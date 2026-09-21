"""
Paper-to-CAD FastMCP Server
Stage 1: Core Tools for converting structured schema definitions into digital engineering formats.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from mcp.server.mcpserver import MCPServer
    FastMCP = MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP

import ezdxf
from ezdxf.enums import TextEntityAlignment
import graphviz
from pydantic import BaseModel, Field

# Initialize FastMCP / MCPServer instance
mcp = FastMCP("paper-to-cad")


def _find_graphviz_dot() -> Optional[str]:
    """Locate Graphviz dot executable in PATH or standard installation paths."""
    dot_path = shutil.which("dot")
    if dot_path:
        return dot_path

    possible_paths = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Graphviz" / "bin" / "dot.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Graphviz" / "bin" / "dot.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Graphviz" / "bin" / "dot.exe",
        Path(r"C:\Graphviz\bin\dot.exe"),
        Path(r"C:\Program Files\Graphviz\bin\dot.exe"),
    ]

    for p in possible_paths:
        if p.is_file():
            bin_dir = str(p.parent)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return str(p)

    return None


def _render_fallback_graphviz(dot_code: str, base_path: Path) -> Tuple[str, str]:
    """
    Fallback renderer when native Graphviz 'dot' binary is unavailable.
    Parses DOT nodes and edges to construct a clean SVG and PNG rendering.
    """
    # Extract node definitions and labels
    # e.g.: node_id [label="...", shape="..."]
    node_pattern = re.compile(r'([A-Za-z0-9_]+)\s*(?:\[([^\]]*)\])?')
    edge_pattern = re.compile(r'([A-Za-z0-9_]+)\s*(?:->|--)\s*([A-Za-z0-9_]+)')

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Tuple[str, str]] = []

    # Clean lines and extract statements
    for line in dot_code.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#") or "{" in line or "}" in line:
            continue
        line = line.rstrip(";").strip()

        # Check edge first
        edge_match = edge_pattern.search(line)
        if edge_match:
            u, v = edge_match.group(1), edge_match.group(2)
            edges.append((u, v))
            if u not in nodes:
                nodes[u] = {"label": u, "shape": "box"}
            if v not in nodes:
                nodes[v] = {"label": v, "shape": "box"}
            continue

        # Check node
        node_match = node_pattern.match(line)
        if node_match:
            nid = node_match.group(1)
            attrs_str = node_match.group(2) or ""
            if nid.lower() in (
                "graph", "digraph", "subgraph", "node", "edge", "strict",
                "rankdir", "nodesep", "ranksep", "splines", "ratio", "size", "fontname"
            ):
                continue
            if "=" in line and not attrs_str:
                # Graph/layout assignment like rankdir=TB
                continue

            label = nid
            shape = "box"
            if attrs_str:
                label_m = re.search(r'label\s*=\s*["\']?([^"\',\]]+)["\']?', attrs_str)
                if label_m:
                    label = label_m.group(1).strip()
                shape_m = re.search(r'shape\s*=\s*["\']?([a-zA-Z]+)["\']?', attrs_str)
                if shape_m:
                    shape = shape_m.group(1).strip()

            nodes[nid] = {"label": label, "shape": shape}

    # Layout nodes top-to-bottom
    node_ids = list(nodes.keys())
    spacing_y = 90
    spacing_x = 180
    svg_width = max(400, spacing_x * 2 + 100)
    svg_height = max(300, (len(node_ids) + 1) * spacing_y)

    positions: Dict[str, Tuple[int, int]] = {}
    center_x = svg_width // 2
    for idx, nid in enumerate(node_ids):
        positions[nid] = (center_x, 60 + idx * spacing_y)

    # Build SVG content
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <defs>',
        '    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1e293b" />',
        '    </marker>',
        '    <style>',
        '      .node-box { fill: #f8fafc; stroke: #0284c7; stroke-width: 2; rx: 6; }',
        '      .node-circle { fill: #f8fafc; stroke: #10b981; stroke-width: 2; }',
        '      .node-text { font-family: sans-serif; font-size: 13px; font-weight: 600; fill: #0f172a; text-anchor: middle; dominant-baseline: central; }',
        '      .edge-line { stroke: #1e293b; stroke-width: 2; marker-end: url(#arrow); }',
        '    </style>',
        '  </defs>',
        f'  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" />',
    ]

    # Draw edges
    for u, v in edges:
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            # Offset start and end slightly for node boundaries
            if y2 > y1:
                start_y = y1 + 22
                end_y = y2 - 22
            else:
                start_y = y1 - 22
                end_y = y2 + 22
            svg_lines.append(f'  <line x1="{x1}" y1="{start_y}" x2="{x2}" y2="{end_y}" class="edge-line" />')

    # Draw nodes
    for nid, data in nodes.items():
        x, y = positions[nid]
        lbl_display = data.get("label", nid).replace("\\n", "\n")
        shape = data.get("shape", "box")
        lines_list = lbl_display.split("\n")
        if shape in ("circle", "ellipse", "doublecircle"):
            r = 28
            svg_lines.append(f'  <circle cx="{x}" cy="{y}" r="{r}" class="node-circle" />')
        else:
            w, h = 110, 48
            svg_lines.append(f'  <rect x="{x - w//2}" y="{y - h//2}" width="{w}" height="{h}" class="node-box" />')

        if len(lines_list) == 1:
            svg_lines.append(f'  <text x="{x}" y="{y}" class="node-text">{lines_list[0]}</text>')
        else:
            line_height = 14
            start_y = y - ((len(lines_list) - 1) * line_height) // 2
            svg_lines.append(f'  <text x="{x}" y="{start_y}" class="node-text">')
            for i, line_txt in enumerate(lines_list):
                dy_attr = f' dy="{line_height}px"' if i > 0 else ''
                svg_lines.append(f'    <tspan x="{x}"{dy_attr}>{line_txt}</tspan>')
            svg_lines.append('  </text>')

    svg_lines.append('</svg>')
    svg_content = "\n".join(svg_lines)

    svg_file = base_path.with_suffix(".svg")
    png_file = base_path.with_suffix(".png")

    svg_file.write_text(svg_content, encoding="utf-8")

    # Generate PNG using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (svg_width, svg_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw edges
        for u, v in edges:
            if u in positions and v in positions:
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                start_y = y1 + 24 if y2 > y1 else y1 - 24
                end_y = y2 - 24 if y2 > y1 else y2 + 24
                draw.line([(x1, start_y), (x2, end_y)], fill=(30, 41, 59), width=2)
                # Arrowhead
                draw.polygon([(x2, end_y), (x2 - 5, end_y - 8), (x2 + 5, end_y - 8)], fill=(30, 41, 59))

        # Draw nodes
        for nid, data in nodes.items():
            x, y = positions[nid]
            lbl_display = data.get("label", nid).replace("\\n", "\n")
            shape = data.get("shape", "box")
            if shape in ("circle", "ellipse", "doublecircle"):
                r = 28
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(248, 250, 252), outline=(16, 185, 129), width=2)
            else:
                w, h = 110, 48
                draw.rectangle([(x - w // 2, y - h // 2), (x + w // 2, y + h // 2)], fill=(248, 250, 252), outline=(2, 132, 199), width=2)

            # Draw text label centered
            bbox = draw.multiline_textbbox((0, 0), lbl_display)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.multiline_text((x - tw // 2, y - th // 2), lbl_display, fill=(15, 23, 42), align="center")

        img.save(png_file, "PNG")
    except Exception as img_err:
        # Fallback 1x1 png placeholder if PIL fails
        if not png_file.exists():
            png_file.write_bytes(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\r\xef\x05f\x00\x00\x00\x00IEND\xaeB`\x82'
            )

    return str(svg_file.resolve()), str(png_file.resolve())


@mcp.tool()
def export_to_graphviz(dot_code: str, output_filename: str) -> str:
    """
    Takes a Graphviz DOT string, compiles it to an SVG and PNG, and saves it locally.

    Args:
        dot_code: Valid Graphviz DOT definition string.
        output_filename: Output base filename or filepath (e.g. 'output/circuit' or 'circuit.svg').

    Returns:
        The absolute path to the generated primary output file (SVG).
    """
    target_path = Path(output_filename)

    # If relative path without explicit parent folder, default to output/
    if len(target_path.parts) == 1:
        target_path = Path("output") / target_path

    # Strip any extension to get the base path
    if target_path.suffix.lower() in (".svg", ".png", ".dot", ".pdf"):
        base_path = target_path.with_suffix("")
    else:
        base_path = target_path

    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DOT source file
    dot_file = base_path.with_suffix(".dot")
    dot_file.write_text(dot_code, encoding="utf-8")

    # Check for dot binary
    dot_bin = _find_graphviz_dot()

    if dot_bin:
        try:
            src = graphviz.Source(dot_code)
            # Render SVG and PNG
            svg_out = src.render(filename=str(base_path), format="svg", cleanup=False)
            png_out = src.render(filename=str(base_path), format="png", cleanup=False)
            return str(Path(svg_out).resolve())
        except Exception:
            # If native render fails for any reason, fall back gracefully
            pass

    # Use fallback renderer
    svg_out, png_out = _render_fallback_graphviz(dot_code, base_path)
    return svg_out


@mcp.tool()
def generate_dxf_schematic(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    filename: str,
) -> str:
    """
    Uses ezdxf to draw clean 2D CAD blocks (rectangles for components,
    lines for wires/traces, and text labels for pin numbers and values).
    Saves as a standard .dxf file and returns the path.

    Args:
        components: List of component dicts with fields:
                    - id: str (e.g. 'R1', 'C1', 'V1')
                    - type: Optional[str] (e.g. 'resistor', 'voltage_source')
                    - value: Optional[str] (e.g. '10k', '5V', '0.1uF')
                    - x: float (X coordinate)
                    - y: float (Y coordinate)
                    - width: Optional[float] (default 40.0)
                    - height: Optional[float] (default 24.0)
                    - pins: Optional[List[Dict[str, Any]]] (e.g. [{'id': '1', 'name': 'IN', 'x': 80, 'y': 100}])
        connections: List of connection dicts with fields:
                     - from: str or Dict (e.g. 'R1:2' or {'component': 'R1', 'pin': '2'})
                     - to: str or Dict (e.g. 'R2:1' or {'component': 'R2', 'pin': '1'})
                     - points: Optional[List[List[float]]] (custom routed coordinates)
                     - label: Optional[str] (net name e.g. 'VOUT')
        filename: Destination file path (e.g. 'output/schematic.dxf').

    Returns:
        The absolute path to the generated .dxf file.
    """
    target_path = Path(filename)
    if len(target_path.parts) == 1:
        target_path = Path("output") / target_path

    if target_path.suffix.lower() != ".dxf":
        target_path = target_path.with_suffix(".dxf")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DXF R2010 document
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create distinct CAD layers
    # Layer colors: 7=White, 1=Red, 3=Green, 2=Yellow, 4=Cyan
    layers = [
        ("COMPONENTS", 7),
        ("PINS", 1),
        ("WIRES", 3),
        ("TEXT", 2),
        ("ANNOTATIONS", 4),
    ]
    for layer_name, color in layers:
        if layer_name not in doc.layers:
            doc.layers.add(name=layer_name, color=color)

    pin_catalog: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # Draw components
    for comp in components:
        cid = str(comp.get("id", comp.get("name", "COMP")))
        ctype = str(comp.get("type", "")).upper()
        cval = str(comp.get("value", ""))
        cx = float(comp.get("x", 0.0))
        cy = float(comp.get("y", 0.0))
        w = float(comp.get("width", 40.0))
        h = float(comp.get("height", 24.0))

        half_w = w / 2.0
        half_h = h / 2.0

        # Component bounding box
        rect_pts = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        msp.add_lwpolyline(rect_pts, close=True, dxfattribs={"layer": "COMPONENTS"})

        # Primary label (Designator e.g. R1)
        msp.add_text(
            cid,
            height=3.2,
            dxfattribs={"layer": "TEXT"},
        ).set_placement((cx, cy + 2.5), align=TextEntityAlignment.MIDDLE_CENTER)

        # Secondary label (Value or Type e.g. 10k)
        sub_label = cval if cval else ctype
        if sub_label:
            msp.add_text(
                sub_label,
                height=2.4,
                dxfattribs={"layer": "TEXT"},
            ).set_placement((cx, cy - 3.5), align=TextEntityAlignment.MIDDLE_CENTER)

        # Pin extraction & placement
        pins_data = comp.get("pins")
        resolved_pins: List[Dict[str, Any]] = []

        if pins_data and isinstance(pins_data, list):
            for p in pins_data:
                if isinstance(p, dict):
                    resolved_pins.append(p)
                else:
                    resolved_pins.append({"id": str(p), "name": str(p)})
        else:
            # Default 2-pin layout (left & right)
            resolved_pins = [
                {"id": "1", "name": "1"},
                {"id": "2", "name": "2"},
            ]

        pin_count = len(resolved_pins)
        stub_len = 6.0

        for i, pin in enumerate(resolved_pins):
            pid = str(pin.get("id", str(i + 1)))
            pname = str(pin.get("name", pid))

            if "x" in pin and "y" in pin:
                term_x = float(pin["x"])
                term_y = float(pin["y"])
                # Stub connected to closest component edge
                edge_x = cx - half_w if term_x <= cx else cx + half_w
                edge_y = term_y
            else:
                # Automatic pin layout based on pin count
                if pin_count == 2:
                    if i == 0:  # Pin 1: Left
                        edge_x, edge_y = cx - half_w, cy
                        term_x, term_y = edge_x - stub_len, edge_y
                    else:  # Pin 2: Right
                        edge_x, edge_y = cx + half_w, cy
                        term_x, term_y = edge_x + stub_len, edge_y
                elif pin_count == 3:
                    if i == 0:  # Top
                        edge_x, edge_y = cx, cy + half_h
                        term_x, term_y = edge_x, edge_y + stub_len
                    elif i == 1:  # Left
                        edge_x, edge_y = cx - half_w, cy
                        term_x, term_y = edge_x - stub_len, edge_y
                    else:  # Right
                        edge_x, edge_y = cx + half_w, cy
                        term_x, term_y = edge_x + stub_len, edge_y
                else:
                    # Generic evenly spaced on left/right edges
                    side_left = i < (pin_count // 2)
                    idx_on_side = i if side_left else i - (pin_count // 2)
                    side_count = (pin_count // 2) if side_left else (pin_count - pin_count // 2)
                    y_offset = (idx_on_side + 0.5) / side_count * h - half_h
                    edge_y = cy + y_offset
                    if side_left:
                        edge_x = cx - half_w
                        term_x = edge_x - stub_len
                    else:
                        edge_x = cx + half_w
                        term_x = edge_x + stub_len
                    term_y = edge_y

            # Draw pin lead stub
            msp.add_line((edge_x, edge_y), (term_x, term_y), dxfattribs={"layer": "PINS"})

            # Draw pin terminal circle
            msp.add_circle((term_x, term_y), radius=0.8, dxfattribs={"layer": "PINS"})

            # Pin text label
            text_x = term_x + (1.2 if term_x >= cx else -1.2)
            text_align = TextEntityAlignment.MIDDLE_LEFT if term_x >= cx else TextEntityAlignment.MIDDLE_RIGHT
            msp.add_text(
                pname,
                height=1.8,
                dxfattribs={"layer": "PINS"},
            ).set_placement((text_x, term_y), align=text_align)

            pin_catalog[(cid, pid)] = (term_x, term_y)
            pin_catalog[(cid, pname)] = (term_x, term_y)

    # Helper to parse connection endpoint
    def parse_endpoint(ep: Any) -> Optional[Tuple[float, float]]:
        if isinstance(ep, (list, tuple)) and len(ep) >= 2:
            return float(ep[0]), float(ep[1])
        if isinstance(ep, dict):
            if "x" in ep and "y" in ep:
                return float(ep["x"]), float(ep["y"])
            comp = str(ep.get("component", ep.get("comp", "")))
            pin = str(ep.get("pin", ""))
            return pin_catalog.get((comp, pin))
        if isinstance(ep, str):
            # Formats: 'R1:1', 'R1.1', 'R1-1', or direct comp ID
            parts = re.split(r'[:.\-]', ep, maxsplit=1)
            if len(parts) == 2:
                return pin_catalog.get((parts[0].strip(), parts[1].strip()))
            elif len(parts) == 1:
                # Try pin 1 or 2 default
                return pin_catalog.get((parts[0].strip(), "1"))
        return None

    # Draw connections
    for conn in connections:
        custom_points = conn.get("points")
        label = conn.get("label")

        if custom_points and isinstance(custom_points, list) and len(custom_points) >= 2:
            wire_pts = [(float(p[0]), float(p[1])) for p in custom_points]
            msp.add_lwpolyline(wire_pts, dxfattribs={"layer": "WIRES"})
            mid_pt = wire_pts[len(wire_pts) // 2]
        else:
            ep_from = parse_endpoint(conn.get("from", conn.get("source")))
            ep_to = parse_endpoint(conn.get("to", conn.get("target")))

            if ep_from and ep_to:
                x1, y1 = ep_from
                x2, y2 = ep_to

                # Orthogonal routing: step in X then Y, or mid-step
                if math.isclose(x1, x2, abs_tol=1e-3) or math.isclose(y1, y2, abs_tol=1e-3):
                    wire_pts = [(x1, y1), (x2, y2)]
                else:
                    mid_x = (x1 + x2) / 2.0
                    wire_pts = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]

                msp.add_lwpolyline(wire_pts, dxfattribs={"layer": "WIRES"})
                mid_pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            else:
                continue

        # Optional Net/Wire Annotation
        if label and mid_pt:
            msp.add_text(
                str(label),
                height=2.0,
                dxfattribs={"layer": "ANNOTATIONS"},
            ).set_placement((mid_pt[0], mid_pt[1] + 1.5), align=TextEntityAlignment.MIDDLE_CENTER)

    doc.saveas(str(target_path))
    return str(target_path.resolve())


@mcp.tool()
def validate_spice_netlist(netlist_text: str) -> Dict[str, Any]:
    """
    Validates the basic syntax of a SPICE netlist string:
    - Checks for standard reference designators (R, C, L, V, I, D, Q, M, J, X, etc.)
    - Verifies the presence of a ground reference node ('0' or 'GND')
    - Detects floating unconnected nodes (nodes referenced by only one terminal)
    - Verifies proper terminal counts for standard passive and active elements

    Args:
        netlist_text: Raw SPICE netlist string.

    Returns:
        Structured dictionary with validation status, component counts, nodes, warnings, and errors.
    """
    errors: List[str] = []
    warnings: List[str] = []
    parsed_components: List[Dict[str, Any]] = []
    directives: List[str] = []

    # Map node -> list of (component_id, pin_index)
    node_connections: Dict[str, List[Tuple[str, int]]] = {}

    # Known reference designators and minimum required terminal counts
    # Designator -> (Full Name, Min Terminals, Has Value/Model)
    spice_device_specs: Dict[str, Tuple[str, int, bool]] = {
        "R": ("Resistor", 2, True),
        "C": ("Capacitor", 2, True),
        "L": ("Inductor", 2, True),
        "V": ("Independent Voltage Source", 2, True),
        "I": ("Independent Current Source", 2, True),
        "D": ("Diode", 2, True),
        "Q": ("BJT Transistor", 3, True),
        "M": ("MOSFET Transistor", 4, True),
        "J": ("JFET Transistor", 3, True),
        "X": ("Subcircuit Instance", 1, True),
        "E": ("Voltage-Controlled Voltage Source", 4, True),
        "F": ("Current-Controlled Current Source", 2, True),
        "G": ("Voltage-Controlled Current Source", 4, True),
        "H": ("Current-Controlled Voltage Source", 2, True),
        "K": ("Coupled Inductor", 2, True),
        "S": ("Voltage-Controlled Switch", 4, True),
        "W": ("Current-Controlled Switch", 2, True),
    }

    # Step 1: Preprocessing & Line continuation
    raw_lines = netlist_text.splitlines()
    merged_lines: List[Tuple[int, str]] = []

    current_line = ""
    current_line_num = 0

    for idx, raw_line in enumerate(raw_lines, start=1):
        # Strip trailing comments (; or $)
        line_clean = re.split(r'[;$]', raw_line, maxsplit=1)[0].strip()
        if not line_clean:
            continue

        if line_clean.startswith("+"):
            # Continuation line
            current_line += " " + line_clean[1:].strip()
        else:
            if current_line:
                merged_lines.append((current_line_num, current_line))
            current_line = line_clean
            current_line_num = idx

    if current_line:
        merged_lines.append((current_line_num, current_line))

    # Step 2: Line Parsing
    has_end_directive = False

    for line_idx, (orig_num, line) in enumerate(merged_lines):
        # First line of a netlist file in classic SPICE is often the title if not a directive/comment
        if line_idx == 0 and not line.startswith("*") and not line.startswith("."):
            # Check if it looks like a component or a title
            first_char = line[0].upper()
            tokens = line.split()
            if first_char not in spice_device_specs or len(tokens) == 1:
                # Treat as title line
                directives.append(f"TITLE: {line}")
                continue

        # Comment line
        if line.startswith("*"):
            continue

        # SPICE Directives (.TRAN, .AC, .OP, .MODEL, .SUBCKT, .END, etc.)
        if line.startswith("."):
            directive_name = line.split()[0].upper()
            directives.append(line)
            if directive_name == ".END":
                has_end_directive = True
            continue

        # Device line
        tokens = line.split()
        if not tokens:
            continue

        comp_name = tokens[0]
        prefix = comp_name[0].upper()

        if prefix not in spice_device_specs:
            warnings.append(
                f"Line {orig_num}: Unrecognized reference designator '{comp_name}'. "
                f"Prefix '{prefix}' is not a standard SPICE device type."
            )
            continue

        device_name, min_terminals, requires_value = spice_device_specs[prefix]

        # Minimum tokens check (comp_name + min_terminals + [value/model])
        min_required_tokens = 1 + min_terminals
        if len(tokens) < min_required_tokens:
            errors.append(
                f"Line {orig_num}: Malformed component '{comp_name}' ({device_name}). "
                f"Expected at least {min_terminals} node terminals, found {len(tokens) - 1} tokens: {tokens[1:]}."
            )
            continue

        # Extract terminals and value/model
        # For subcircuits 'X', the last token is always the subcircuit model name
        if prefix == "X":
            nodes = tokens[1:-1]
            val_model = tokens[-1]
        elif prefix in ("Q", "M", "J", "D"):
            # Semiconductor devices usually have model name as the last token
            nodes = tokens[1 : 1 + min_terminals]
            val_model = " ".join(tokens[1 + min_terminals :]) if len(tokens) > 1 + min_terminals else ""
        else:
            nodes = tokens[1 : 1 + min_terminals]
            val_model = " ".join(tokens[1 + min_terminals :])

        if requires_value and not val_model and prefix not in ("X", "Q", "M"):
            warnings.append(
                f"Line {orig_num}: Component '{comp_name}' is missing an electrical value or model specification."
            )

        # Record component
        parsed_components.append({
            "name": comp_name,
            "type": device_name,
            "nodes": nodes,
            "value": val_model,
            "line": orig_num,
        })

        # Track node usage
        for pin_idx, node in enumerate(nodes, start=1):
            normalized_node = node.upper()
            if normalized_node not in node_connections:
                node_connections[normalized_node] = []
            node_connections[normalized_node].append((comp_name, pin_idx))

    # Step 3: Validate Circuit-Level Integrity
    total_components = len(parsed_components)
    all_nodes = set(node_connections.keys())

    if total_components == 0:
        errors.append("Netlist contains no valid circuit components.")

    # Check ground reference node ('0' or 'GND')
    has_ground = ("0" in all_nodes) or ("GND" in all_nodes)
    if not has_ground and total_components > 0:
        errors.append(
            "Missing ground reference node ('0' or 'GND'). "
            "SPICE simulators require node '0' to be established as the global reference."
        )

    # Check for floating / unconnected nodes (connected to only 1 terminal)
    floating_nodes = []
    for node, connections_list in node_connections.items():
        if len(connections_list) == 1:
            floating_nodes.append(node)
            comp_id, pin_num = connections_list[0]
            warnings.append(
                f"Floating node detected: Node '{node}' is connected only to terminal {pin_num} "
                f"of component '{comp_id}'. Node has no return path."
            )

    # Check for components with all identical nodes (short-circuited to itself)
    for comp in parsed_components:
        comp_nodes = comp["nodes"]
        if len(comp_nodes) >= 2 and len(set(comp_nodes)) == 1:
            warnings.append(
                f"Component '{comp['name']}' has all terminals connected to the same node '{comp_nodes[0]}' "
                f"(self-shorted)."
            )

    is_valid = len(errors) == 0
    status = "VALID" if is_valid and not warnings else ("WARNINGS" if is_valid else "INVALID")

    summary_text = (
        f"Validation status: {status}. Found {total_components} components and {len(all_nodes)} nodes. "
        f"{'Ground node detected.' if has_ground else 'No ground node!'} "
        f"{len(floating_nodes)} floating nodes, {len(warnings)} warnings, {len(errors)} errors."
    )

    return {
        "status": status,
        "is_valid": is_valid,
        "summary": summary_text,
        "components_count": total_components,
        "nodes_count": len(all_nodes),
        "nodes": sorted(list(all_nodes)),
        "floating_nodes": sorted(floating_nodes),
        "has_ground": has_ground,
        "parsed_components": parsed_components,
        "directives": directives,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    # Support direct execution of the FastMCP server
    mcp.run()
