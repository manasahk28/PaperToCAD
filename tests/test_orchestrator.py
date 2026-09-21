"""
Stage 3 Orchestrator and Reasoning Agent Test Suite
Tests:
- Schema validation of SchematicInspectionResult
- inspect_and_convert execution and automatic MCP tool routing
- Tool execution: export_to_graphviz, generate_dxf_schematic, and validate_spice_netlist
- Mock client integration and error propagation
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ezdxf
import pytest
from agent.inspector import (
    ComponentDefinition,
    ConnectionDefinition,
    SchematicInspectionResult,
)
from agent.orchestrator import inspect_and_convert
from tests.test_vision import generate_synthetic_sketch


def test_schematic_inspection_result_schema():
    """Verify that SchematicInspectionResult enforces strict schema validation."""
    data = {
        "diagram_type": "circuit",
        "detected_components": [
            {
                "id": "Vin",
                "name": "Vin 12V",
                "type": "voltage_source",
                "value": "12V",
                "pinout": ["1", "2"],
                "bbox": [100, 200, 160, 90],
                "center": [180, 245],
            },
            {
                "id": "R1",
                "name": "R1 10k",
                "type": "resistor",
                "value": "10k",
                "pinout": ["1", "2"],
                "bbox": [370, 200, 160, 90],
                "center": [450, 245],
            },
        ],
        "connections": [
            {"source": "Vin:2", "target": "R1:1", "label": "NET_VIN"}
        ],
        "design_errors": ["Missing ground reference"],
        "suggested_fix": "Add ground reference node 0 to return line.",
        "mcp_tool_payload": "V1 IN 0 DC 12V\nR1 IN OUT 10k\n.END",
        "target_mcp_tool": "validate_spice_netlist",
    }

    result = SchematicInspectionResult.model_validate(data)
    assert result.diagram_type == "circuit"
    assert len(result.detected_components) == 2
    assert result.detected_components[0].id == "Vin"
    assert len(result.design_errors) == 1
    assert result.target_mcp_tool == "validate_spice_netlist"

    # Verify JSON serialization round-trip
    json_str = result.model_dump_json()
    reloaded = SchematicInspectionResult.model_validate_json(json_str)
    assert reloaded == result
    print("[PASS] test_schematic_inspection_result_schema: Validated successfully.")


def test_inspect_and_convert_circuit():
    """Verify orchestrator inspects a circuit sketch and routes to validate_spice_netlist + DXF."""
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "orchestrator_test_circuit.png"
    generate_synthetic_sketch(output_path=img_path)

    cv_data = {
        "raw_texts": ["Vin 12V", "R1 10k", "Vout 5V", "GND"],
        "text_detections": [
            {"text": "Vin 12V", "center": [180, 245], "rect": [122, 220, 100, 30]},
            {"text": "R1 10k", "center": [450, 245], "rect": [398, 220, 90, 30]},
            {"text": "Vout 5V", "center": [720, 245], "rect": [662, 220, 100, 30]},
            {"text": "GND", "center": [580, 410], "rect": [535, 395, 60, 25]},
        ],
        "shape_detections": [
            {"type": "box", "bbox": [100, 200, 160, 90], "center": [180, 245]},
            {"type": "box", "bbox": [370, 200, 160, 90], "center": [450, 245]},
            {"type": "box", "bbox": [640, 200, 160, 90], "center": [720, 245]},
            {"type": "junction", "bbox": [572, 237, 16, 16], "center": [580, 245], "radius": 8.0},
        ],
    }

    result = inspect_and_convert(str(img_path), cv_data)

    assert isinstance(result, dict)
    assert "inspection" in result
    assert "routed_tool" in result
    assert "tool_output" in result
    assert "generated_files" in result
    assert result["routed_tool"] == "validate_spice_netlist"

    # Verify SPICE validation report
    spice_report = result["tool_output"]
    assert isinstance(spice_report, dict)
    assert "is_valid" in spice_report
    assert spice_report["is_valid"] is True
    assert spice_report["has_ground"] is True

    # Verify DXF CAD file was generated
    assert len(result["generated_files"]) > 0
    dxf_file = Path(result["generated_files"][0])
    assert dxf_file.exists()
    assert dxf_file.suffix == ".dxf"

    # Inspect DXF with ezdxf
    doc = ezdxf.readfile(str(dxf_file))
    assert len(list(doc.modelspace())) > 0

    print(
        f"[PASS] test_inspect_and_convert_circuit: Successfully audited and generated DXF at {dxf_file} "
        f"with SPICE status: {spice_report['status']}"
    )


def test_inspect_and_convert_flowchart():
    """Verify orchestrator inspects a flowchart sketch and routes to export_to_graphviz."""
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "orchestrator_test_flowchart.png"
    generate_synthetic_sketch(output_path=img_path)

    cv_data = {
        "raw_texts": ["START PROCESS", "VALIDATE INPUT", "GENERATE CAD"],
        "text_detections": [
            {"text": "START PROCESS", "center": [180, 245], "rect": [120, 220, 120, 30]},
            {"text": "VALIDATE INPUT", "center": [450, 245], "rect": [390, 220, 120, 30]},
            {"text": "GENERATE CAD", "center": [720, 245], "rect": [660, 220, 120, 30]},
        ],
        "shape_detections": [
            {"type": "box", "bbox": [100, 200, 160, 90], "center": [180, 245]},
            {"type": "box", "bbox": [370, 200, 160, 90], "center": [450, 245]},
            {"type": "box", "bbox": [640, 200, 160, 90], "center": [720, 245]},
        ],
    }

    result = inspect_and_convert(str(img_path), cv_data)

    assert result["routed_tool"] == "export_to_graphviz"
    assert len(result["generated_files"]) > 0

    svg_file = Path(result["generated_files"][0])
    assert svg_file.exists()
    assert svg_file.suffix == ".svg"
    assert "<svg" in svg_file.read_text(encoding="utf-8")

    print(f"[PASS] test_inspect_and_convert_flowchart: Generated SVG flowchart at {svg_file}")


def test_orchestrator_with_custom_client():
    """Verify orchestrator respects custom client output and executes tools correctly."""
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "orchestrator_custom.png"
    generate_synthetic_sketch(output_path=img_path)

    class MockInspectorClient:
        def inspect(self, image_path, cv_data):
            return SchematicInspectionResult(
                diagram_type="circuit",
                detected_components=[
                    ComponentDefinition(id="R1", name="R1", type="resistor", value="1k", bbox=[10, 10, 40, 20]),
                    ComponentDefinition(id="R2", name="R2", type="resistor", value="2k", bbox=[60, 10, 40, 20]),
                ],
                connections=[
                    ConnectionDefinition(source="R1:2", target="R2:1", label="MID")
                ],
                design_errors=["Floating node R2 pin 2", "Missing ground reference"],
                suggested_fix="Connect ground to R2 pin 2.",
                mcp_tool_payload="digraph Circuit { R1 -> R2; }",
                target_mcp_tool="export_to_graphviz",
            )

    mock_client = MockInspectorClient()
    result = inspect_and_convert(str(img_path), {}, client=mock_client)

    assert result["status"] == "WARNING"  # Flagged design errors
    assert result["routed_tool"] == "export_to_graphviz"
    assert len(result["inspection"]["design_errors"]) == 2
    assert len(result["generated_files"]) > 0
    print("[PASS] test_orchestrator_with_custom_client: Successfully verified mock client.")


def main():
    """Run all tests directly without pytest."""
    print("Running Stage 3 Orchestrator & Reasoning Agent test suite...\n" + "=" * 60)
    test_schematic_inspection_result_schema()
    test_inspect_and_convert_circuit()
    test_inspect_and_convert_flowchart()
    test_orchestrator_with_custom_client()
    print("=" * 60 + "\nAll Stage 3 orchestrator tests passed successfully!")


if __name__ == "__main__":
    main()
