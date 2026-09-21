"""
Standalone Test Suite for Paper-to-CAD FastMCP Tools
Tests:
- export_to_graphviz: Generates SVG/PNG/DOT files for a voltage divider.
- generate_dxf_schematic: Generates DXF CAD schematic with CAD entities.
- validate_spice_netlist: Tests valid circuits, floating nodes, and syntax errors.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so server module can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ezdxf
import pytest
from server.mcp_server import (
    export_to_graphviz,
    generate_dxf_schematic,
    validate_spice_netlist,
)


def test_export_to_graphviz():
    """Test export_to_graphviz creates valid SVG, PNG, and DOT files."""
    dot_code = """
    digraph VoltageDivider {
        rankdir=TB;
        node [shape=box, style=rounded, fontname="Helvetica"];
        Vin [shape=circle, label="Vin\\n12V"];
        R1 [shape=box, label="R1\\n10k"];
        R2 [shape=box, label="R2\\n4.7k"];
        GND [shape=circle, label="GND\\n0V"];

        Vin -> R1 [label="V_in"];
        R1 -> R2 [label="V_out"];
        R2 -> GND;
    }
    """
    output_filename = "output/test_voltage_divider.svg"
    result_path = export_to_graphviz(dot_code, output_filename)

    svg_file = Path(result_path)
    base_file = svg_file.with_suffix("")
    dot_file = base_file.with_suffix(".dot")
    png_file = base_file.with_suffix(".png")

    assert svg_file.exists(), f"SVG file was not generated at {svg_file}"
    assert svg_file.stat().st_size > 0, "SVG file is empty"

    assert dot_file.exists(), f"DOT file was not generated at {dot_file}"
    assert dot_file.stat().st_size > 0, "DOT file is empty"

    assert png_file.exists(), f"PNG file was not generated at {png_file}"
    assert png_file.stat().st_size > 0, "PNG file is empty"

    # Verify SVG contains XML/SVG tags
    content = svg_file.read_text(encoding="utf-8", errors="ignore")
    assert "<svg" in content and "</svg>" in content, "Invalid SVG format"
    print(f"[PASS] test_export_to_graphviz: Generated {svg_file} ({svg_file.stat().st_size} bytes)")


def test_generate_dxf_schematic():
    """Test generate_dxf_schematic creates a valid DXF with components, pins, and wires."""
    components = [
        {
            "id": "Vin",
            "type": "voltage_source",
            "value": "12V",
            "x": 60.0,
            "y": 120.0,
            "width": 36.0,
            "height": 24.0,
            "pins": [
                {"id": "1", "name": "-", "x": 60.0, "y": 96.0},
                {"id": "2", "name": "+", "x": 60.0, "y": 144.0},
            ],
        },
        {
            "id": "R1",
            "type": "resistor",
            "value": "10k",
            "x": 160.0,
            "y": 144.0,
            "width": 40.0,
            "height": 20.0,
        },
        {
            "id": "R2",
            "type": "resistor",
            "value": "4.7k",
            "x": 260.0,
            "y": 144.0,
            "width": 40.0,
            "height": 20.0,
        },
        {
            "id": "GND",
            "type": "ground",
            "value": "0V",
            "x": 340.0,
            "y": 144.0,
            "width": 24.0,
            "height": 20.0,
        },
    ]

    connections = [
        {"from": "Vin:2", "to": "R1:1", "label": "NET_VIN"},
        {"from": "R1:2", "to": "R2:1", "label": "V_OUT"},
        {"from": "R2:2", "to": "GND:1", "label": "NET_GND"},
    ]

    filename = "output/test_voltage_divider.dxf"
    result_path = generate_dxf_schematic(components, connections, filename)

    dxf_path = Path(result_path)
    assert dxf_path.exists(), f"DXF file was not generated at {dxf_path}"
    assert dxf_path.stat().st_size > 0, "DXF file is empty"

    # Validate DXF content using ezdxf
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Check layers exist
    expected_layers = {"COMPONENTS", "PINS", "WIRES", "TEXT", "ANNOTATIONS"}
    existing_layers = {layer.dxf.name for layer in doc.layers}
    for l in expected_layers:
        assert l in existing_layers, f"Layer {l} missing in DXF document"

    # Check that entities were added
    entities = list(msp)
    assert len(entities) > 0, "Modelspace contains no entities"

    types = {e.dxftype() for e in entities}
    assert "LWPOLYLINE" in types, "Modelspace missing LWPOLYLINE (rectangles/wires)"
    assert "TEXT" in types, "Modelspace missing TEXT labels"
    assert "CIRCLE" in types, "Modelspace missing CIRCLE pin terminals"

    print(f"[PASS] test_generate_dxf_schematic: Created valid DXF at {dxf_path} with {len(entities)} CAD entities.")


def test_validate_spice_netlist_valid():
    """Test validation of a well-formed voltage divider netlist."""
    netlist = """
    * Voltage Divider Circuit
    V1 IN 0 DC 12V
    R1 IN OUT 10k
    R2 OUT 0 4.7k
    .OP
    .END
    """
    result = validate_spice_netlist(netlist)

    assert result["is_valid"] is True, f"Expected valid netlist, got errors: {result['errors']}"
    assert result["status"] == "VALID"
    assert result["components_count"] == 3
    assert result["has_ground"] is True
    assert set(result["nodes"]) == {"0", "IN", "OUT"}
    assert len(result["floating_nodes"]) == 0
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0
    print("[PASS] test_validate_spice_netlist_valid: Verified correctly.")


def test_validate_spice_netlist_floating_node():
    """Test detection of floating nodes in a netlist."""
    netlist = """
    * Voltage Divider with Floating Terminal
    V1 IN 0 DC 12V
    R1 IN OUT 10k
    R2 FLOAT_NET 0 4.7k
    .END
    """
    result = validate_spice_netlist(netlist)

    assert result["is_valid"] is True  # No fatal syntax error, but has warnings
    assert result["status"] == "WARNINGS"
    assert "FLOAT_NET" in result["floating_nodes"]
    assert "OUT" in result["floating_nodes"]
    assert len(result["warnings"]) >= 2
    print(f"[PASS] test_validate_spice_netlist_floating_node: Detected floating nodes {result['floating_nodes']}.")


def test_validate_spice_netlist_missing_ground():
    """Test netlist missing reference ground node '0'."""
    netlist = """
    * Circuit without Ground
    V1 IN N1 DC 12V
    R1 IN OUT 10k
    R2 OUT N1 4.7k
    .END
    """
    result = validate_spice_netlist(netlist)

    assert result["is_valid"] is False
    assert result["status"] == "INVALID"
    assert result["has_ground"] is False
    assert any("ground" in err.lower() for err in result["errors"])
    print("[PASS] test_validate_spice_netlist_missing_ground: Correctly flagged missing ground.")


def test_validate_spice_netlist_malformed_syntax():
    """Test detection of malformed component syntax."""
    netlist = """
    * Malformed Component
    R1 IN
    .END
    """
    result = validate_spice_netlist(netlist)

    assert result["is_valid"] is False
    assert result["status"] == "INVALID"
    assert any("malformed" in err.lower() for err in result["errors"])
    print("[PASS] test_validate_spice_netlist_malformed_syntax: Correctly flagged syntax error.")


@pytest.mark.anyio
async def test_mcp_server_protocol_calls():
    """Test that all three tools are registered and callable via MCP Server protocol."""
    import asyncio
    from server.mcp_server import mcp

    # Check tools listing
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "export_to_graphviz" in tool_names
    assert "generate_dxf_schematic" in tool_names
    assert "validate_spice_netlist" in tool_names

    # Test invoking validate_spice_netlist via mcp.call_tool
    netlist = """
    * Simple Divider
    V1 IN 0 DC 5V
    R1 IN OUT 2k
    R2 OUT 0 2k
    .END
    """
    call_result = await mcp.call_tool("validate_spice_netlist", {"netlist_text": netlist})
    assert call_result.is_error is False
    assert len(call_result.content) > 0
    assert "VALID" in call_result.content[0].text
    print(f"[PASS] test_mcp_server_protocol_calls: Verified {len(tool_names)} tools via MCP interface.")


def main():
    """Run all tests directly without pytest if invoked as standalone script."""
    import asyncio

    print("Running Stage 1 MCP tools test suite...\n" + "=" * 50)
    test_export_to_graphviz()
    test_generate_dxf_schematic()
    test_validate_spice_netlist_valid()
    test_validate_spice_netlist_floating_node()
    test_validate_spice_netlist_missing_ground()
    test_validate_spice_netlist_malformed_syntax()
    asyncio.run(test_mcp_server_protocol_calls())
    print("=" * 50 + "\nAll 7 tests passed successfully!")


if __name__ == "__main__":
    main()
