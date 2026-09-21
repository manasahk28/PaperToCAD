"""
Multimodal LLM Reasoning Agent & MCP Tool Orchestrator
Stage 3: Paper-to-CAD Interactive Schematic Inspector
Reconciles CV/OCR detections with engineering intent, audits schematics for errors,
and routes payloads to Stage 1 MCP tools.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from agent.inspector import (
    ComponentDefinition,
    ConnectionDefinition,
    SchematicInspectionResult,
)
from server.mcp_server import (
    export_to_graphviz,
    generate_dxf_schematic,
    validate_spice_netlist,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior Principal Electrical & Systems Engineering Inspector.
Your mission is to examine a hand-drawn sketch (circuit schematic, flowchart, or block diagram)
along with preliminary computer vision findings (OCR detected text tokens, geometric bounding boxes, and junction dots).

Tasks:
1. Reconcile OCR tokens (e.g., 'Vin 12V', 'R1 10k', 'Vout 5V', 'GND') with detected boxes/shapes to form coherent components.
2. Audit design integrity:
   - For circuits: Identify missing ground reference ('0' or 'GND'), floating unconnected pins, dangling branches, or short-circuits.
   - For flowcharts: Identify dead-end terminal steps, missing condition branches, or unclosed loops.
3. Suggest clear, actionable engineering fixes.
4. Prepare the exact downstream MCP tool payload:
   - If circuit: generate standard SPICE netlist text or DXF component coordinates.
   - If flowchart or block diagram: generate clean Graphviz DOT code (`digraph G { ... }`).
5. Output MUST be valid JSON adhering strictly to the SchematicInspectionResult schema.
"""


def _heuristic_inspect(
    image_path: Union[str, Path],
    cv_data: Dict[str, Any],
) -> SchematicInspectionResult:
    """
    Intelligent heuristic reasoning inspector.
    Runs when external LLM APIs are unreachable or offline, ensuring deterministic
    reconciliation of CV/OCR findings and strict engineering rule auditing.
    """
    raw_texts = cv_data.get("raw_texts", [])
    text_detections = cv_data.get("text_detections", [])
    shape_detections = cv_data.get("shape_detections", [])

    # Classify diagram type based on tokens
    circuit_patterns = [
        r"\b[rRcCvViIdDqQmM]\d+\b",                         # R1, C1, V1, D1, Q1
        r"\b\d+(\.\d+)?\s*(k|m|u|n|p)?(ohm|v|a|f|h|hz)\b", # 10k, 12V, 5V
        r"\b(gnd|ground|vcc|vdd|vss|vin|vout|anode|cathode)\b",
        r"\b(resistor|capacitor|inductor|diode|transistor|opamp)\b",
    ]
    all_text_combined = " ".join(raw_texts)
    is_circuit = any(re.search(p, all_text_combined, re.IGNORECASE) for p in circuit_patterns)
    diagram_type = "circuit" if is_circuit else "flowchart"

    # Separate boxes and junction dots
    boxes = [s for s in shape_detections if s.get("type") in ("box", "rectangle", "polygon")]
    junctions = [s for s in shape_detections if s.get("type") == "junction"]

    # Reconcile text with closest boxes
    components: List[ComponentDefinition] = []
    used_text_indices = set()

    for idx, box in enumerate(boxes):
        bx, by, bw, bh = box["bbox"]
        bcx, bcy = box["center"]

        # Find overlapping or nearest text
        matched_text = []
        for t_idx, t_item in enumerate(text_detections):
            tcx, tcy = t_item["center"]
            if (bx - 10 <= tcx <= bx + bw + 10) and (by - 10 <= tcy <= by + bh + 10):
                matched_text.append(t_item["text"])
                used_text_indices.add(t_idx)

        comp_label = " ".join(matched_text) if matched_text else f"Node_{idx + 1}"

        # Determine type and value
        comp_type = "block"
        val = None
        cid = f"U{idx + 1}"

        if is_circuit:
            label_lower = comp_label.lower()
            if "v" in label_lower and any(c.isdigit() for c in label_lower) and "r" not in label_lower:
                comp_type = "voltage_source"
                cid = "Vin"
                val = comp_label
            elif "r" in label_lower:
                comp_type = "resistor"
                cid = "R1" if idx == 1 else f"R{idx}"
                val = comp_label
            elif "gnd" in label_lower or "0v" in label_lower:
                comp_type = "ground"
                cid = "GND"
                val = "0V"
            else:
                comp_type = "component"
                cid = f"C_{idx + 1}"
                val = comp_label
        else:
            comp_type = "step"
            cid = f"Step_{idx + 1}"
            val = comp_label

        components.append(ComponentDefinition(
            id=cid,
            name=comp_label,
            type=comp_type,
            value=val,
            pinout=["1", "2"] if is_circuit else ["in", "out"],
            bbox=[bx, by, bw, bh],
            center=[bcx, bcy],
        ))

    # Any remaining text labels (e.g. GND branch)
    for t_idx, t_item in enumerate(text_detections):
        if t_idx not in used_text_indices:
            txt = t_item["text"]
            txt_lower = txt.lower()
            if "gnd" in txt_lower or "0v" in txt_lower:
                components.append(ComponentDefinition(
                    id="GND",
                    name="GND",
                    type="ground",
                    value="0V",
                    pinout=["1"],
                    bbox=t_item.get("rect"),
                    center=t_item.get("center"),
                ))

    # Inferred connections
    connections: List[ConnectionDefinition] = []
    design_errors: List[str] = []
    suggested_fixes: List[str] = []

    if len(components) >= 2:
        # Sort components left to right
        sorted_comps = sorted(components, key=lambda c: c.center[0] if c.center else 0)
        for i in range(len(sorted_comps) - 1):
            c_from = sorted_comps[i]
            c_to = sorted_comps[i + 1]
            connections.append(ConnectionDefinition(
                source=f"{c_from.id}:2" if is_circuit else c_from.id,
                target=f"{c_to.id}:1" if is_circuit else c_to.id,
                label=f"NET_{i+1}",
            ))

    # Auditing engineering rules
    if is_circuit:
        has_ground = any(c.type == "ground" or "gnd" in c.id.lower() or c.id == "0" for c in components)
        if not has_ground:
            design_errors.append("Missing ground reference node ('0' or 'GND'). SPICE simulation requires a zero-potential node.")
            suggested_fixes.append("Establish node 0/GND on the circuit return trace.")

        if junctions:
            # Wire junction exists
            suggested_fixes.append(f"Connect branch at junction ({junctions[0]['center'][0]}, {junctions[0]['center'][1]}) to ground return.")
    else:
        if len(components) > 1 and len(connections) < len(components) - 1:
            design_errors.append("Dead-end flowchart step detected without downstream branching.")
            suggested_fixes.append("Add outgoing condition connector to dead-end step.")

    suggested_fix = " ".join(suggested_fixes) if suggested_fixes else "Schematic is clean and well-formed."

    # Formulate MCP payload
    if is_circuit:
        # Generate clean SPICE netlist
        netlist_lines = ["* Paper-to-CAD Reconciled Schematic Netlist"]
        nodes_created = []
        for i, c in enumerate(components):
            c_val = c.value or "1k"
            val_clean = re.sub(r'[^0-9.kMmuVp]', '', c_val) or "1k"
            if c.type == "voltage_source" or c.id.startswith("V"):
                netlist_lines.append(f"V1 IN 0 DC {val_clean if 'V' in val_clean else val_clean + 'V'}")
            elif c.type == "resistor" or c.id.startswith("R"):
                netlist_lines.append(f"{c.id} IN OUT {val_clean}")
            elif c.type == "ground":
                continue

        # If has load or second resistor
        if len(components) >= 3 and not any("R2" in line for line in netlist_lines):
            netlist_lines.append("R2 OUT 0 4.7k")

        netlist_lines.append(".OP")
        netlist_lines.append(".END")
        mcp_payload = "\n".join(netlist_lines)
        target_tool = "validate_spice_netlist"
    else:
        # Generate DOT format
        dot_lines = ["digraph Flowchart {", "  rankdir=TB;", "  node [shape=box, style=rounded, fontname=\"Helvetica\"];"]
        for c in components:
            dot_lines.append(f'  {c.id} [label="{c.name or c.id}"];')
        for conn in connections:
            dot_lines.append(f"  {conn.source} -> {conn.target};")
        dot_lines.append("}")
        mcp_payload = "\n".join(dot_lines)
        target_tool = "export_to_graphviz"

    return SchematicInspectionResult(
        diagram_type=diagram_type,
        detected_components=components,
        connections=connections,
        design_errors=design_errors,
        suggested_fix=suggested_fix,
        mcp_tool_payload=mcp_payload,
        target_mcp_tool=target_tool,
    )


def _call_gemini(
    image_path: Union[str, Path],
    cv_data: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> Optional[SchematicInspectionResult]:
    """Calls Google Gemini 2.0 Flash via official google-genai SDK."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        img_bytes = Path(image_path).read_bytes()

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Here is the Computer Vision and OCR telemetry extracted from the sketch:\n"
            f"{json.dumps(cv_data, indent=2)}\n\n"
            f"Please inspect the diagram, reconcile findings, audit for errors, and output JSON conforming to the schema."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SchematicInspectionResult,
                temperature=0.1,
            ),
        )

        if response.text:
            return SchematicInspectionResult.model_validate_json(response.text)
    except Exception as e:
        logger.warning(f"Google GenAI call failed: {e}")
    return None


def _call_ollama(
    image_path: Union[str, Path],
    cv_data: Dict[str, Any],
    model_name: str = "qwen2.5-vl",
) -> Optional[SchematicInspectionResult]:
    """Calls local Ollama vision model."""
    try:
        import ollama

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Computer Vision & OCR Telemetry:\n{json.dumps(cv_data, indent=2)}\n\n"
            f"Output JSON conforming to the SchematicInspectionResult schema."
        )

        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt_text,
                "images": [str(image_path)],
            }],
            format=SchematicInspectionResult.model_json_schema(),
        )

        msg_content = response.get("message", {}).get("content", "")
        if msg_content:
            return SchematicInspectionResult.model_validate_json(msg_content)
    except Exception as e:
        logger.debug(f"Ollama call skipped or failed: {e}")
    return None


def inspect_and_convert(
    image_path: Union[str, Path],
    cv_data: Dict[str, Any],
    client: Optional[Any] = None,
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    Multimodal reasoning orchestrator:
    1. Sends the raw sketch image + OCR/CV data to the multimodal LLM inspector.
    2. Reconciles components, labels, and connections, and catches schematic errors.
    3. Automatically routes the output payload to the appropriate Stage 1 MCP tool
       (export_to_graphviz, generate_dxf_schematic, or validate_spice_netlist).

    Args:
        image_path: Path to the hand-drawn sketch image file.
        cv_data: Output dictionary containing OCR text_detections and shape_detections from Stage 2.
        client: Optional pre-configured client or mock for testing.
        model: Model name ('gemini-2.0-flash', 'qwen2.5-vl', etc.).

    Returns:
        Structured orchestration dictionary with inspection results, routed tool, and tool output.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Check for custom/mock client first
    inspection: Optional[SchematicInspectionResult] = None
    if client is not None:
        try:
            if hasattr(client, "inspect"):
                inspection = client.inspect(image_path, cv_data)
            elif hasattr(client, "models"):
                # Custom google-genai client
                res = client.models.generate_content(
                    model=model,
                    contents=[img_path.read_bytes(), json.dumps(cv_data)],
                )
                if res.text:
                    inspection = SchematicInspectionResult.model_validate_json(res.text)
        except Exception as err:
            logger.warning(f"Client inspection error: {err}")

    # If not obtained from custom client, attempt Gemini 2.0 Flash
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if inspection is None and api_key:
        inspection = _call_gemini(img_path, cv_data, api_key=api_key, model_name=model)

    # If still not obtained, attempt local Ollama
    if inspection is None:
        inspection = _call_ollama(img_path, cv_data, model_name="qwen2.5-vl")

    # Fallback to intelligent engineering heuristic inspector
    if inspection is None:
        inspection = _heuristic_inspect(img_path, cv_data)

    # Route payload to appropriate Stage 1 MCP tool
    payload = inspection.mcp_tool_payload.strip()
    target_tool = inspection.target_mcp_tool
    tool_output: Any = None
    generated_files: List[str] = []
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # Automatic tool resolution if target_mcp_tool is unspecified
    if not target_tool:
        if payload.startswith("digraph") or payload.startswith("graph"):
            target_tool = "export_to_graphviz"
        elif any(payload.startswith(p) for p in ("*", ".", "V", "R", "C")):
            target_tool = "validate_spice_netlist"
        else:
            target_tool = "generate_dxf_schematic"

    # Execute designated MCP Tool
    if target_tool == "export_to_graphviz":
        out_svg = out_dir / f"{stem}_mcp_graph.svg"
        tool_output = export_to_graphviz(payload, str(out_svg))
        generated_files.append(tool_output)
        png_path = Path(tool_output).with_suffix(".png")
        if png_path.exists():
            generated_files.append(str(png_path.resolve()))

    elif target_tool == "validate_spice_netlist":
        tool_output = validate_spice_netlist(payload)
        # In addition to SPICE validation, also generate the CAD DXF schematic!
        dxf_components = [c.model_dump() for c in inspection.detected_components]
        dxf_connections = [conn.model_dump() for conn in inspection.connections]
        dxf_path = out_dir / f"{stem}_orchestrated.dxf"
        dxf_file = generate_dxf_schematic(dxf_components, dxf_connections, str(dxf_path))
        generated_files.append(dxf_file)

    elif target_tool == "generate_dxf_schematic":
        try:
            parsed = json.loads(payload)
            comps = parsed.get("components", [c.model_dump() for c in inspection.detected_components])
            conns = parsed.get("connections", [conn.model_dump() for conn in inspection.connections])
        except Exception:
            comps = [c.model_dump() for c in inspection.detected_components]
            conns = [conn.model_dump() for conn in inspection.connections]

        dxf_path = out_dir / f"{stem}_orchestrated.dxf"
        tool_output = generate_dxf_schematic(comps, conns, str(dxf_path))
        generated_files.append(tool_output)

    status = "WARNING" if inspection.design_errors else "SUCCESS"
    summary_msg = (
        f"Inspected '{img_path.name}' as {inspection.diagram_type.upper()}. "
        f"Identified {len(inspection.detected_components)} components and {len(inspection.connections)} connections. "
        f"Flagged {len(inspection.design_errors)} design issues. "
        f"Routed to MCP tool '{target_tool}'."
    )

    return {
        "status": status,
        "inspection": inspection.model_dump(),
        "routed_tool": target_tool,
        "tool_output": tool_output,
        "generated_files": generated_files,
        "summary": summary_msg,
    }
