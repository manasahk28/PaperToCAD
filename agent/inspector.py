"""
Inspector Schema Definitions for Paper-to-CAD Multimodal Reasoning Agent
Stage 3: LLM Reasoning Agent & MCP Tool Execution
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ComponentDefinition(BaseModel):
    """Represents a component or block detected in a schematic sketch."""

    id: str = Field(
        description="Unique identifier, e.g., 'R1', 'V1', 'U1', 'Step1'"
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable label or designator, e.g. 'R1', 'Process Order'"
    )
    type: str = Field(
        description="Component or block type, e.g. 'resistor', 'voltage_source', 'capacitor', 'ground', 'step', 'decision'"
    )
    value: Optional[str] = Field(
        default=None,
        description="Electrical value or action text, e.g. '10k', '12V', '0.1uF'"
    )
    pinout: List[str] = Field(
        default_factory=list,
        description="Inferred terminal or pin names, e.g. ['1', '2'], ['IN', 'OUT'], ['+', '-']"
    )
    bbox: Optional[List[int]] = Field(
        default=None,
        description="Bounding box coordinates in pixels [x, y, w, h]"
    )
    center: Optional[List[int]] = Field(
        default=None,
        description="Centroid point [x, y]"
    )


class ConnectionDefinition(BaseModel):
    """Represents a wire, net, or flow connection between components."""

    source: str = Field(
        description="Origin pin or component ID, e.g., 'Vin:2', 'Vin', 'Step1'"
    )
    target: str = Field(
        description="Destination pin or component ID, e.g., 'R1:1', 'R1', 'Step2'"
    )
    label: Optional[str] = Field(
        default=None,
        description="Signal net name or branch label, e.g., 'V_OUT', 'YES', 'NET1'"
    )
    points: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional waypoint coordinates for routing [[x1, y1], [x2, y2]]"
    )


class SchematicInspectionResult(BaseModel):
    """Structured audit and conversion result produced by the Multimodal LLM reasoning agent."""

    diagram_type: Literal["circuit", "flowchart", "block_diagram", "pid"] = Field(
        description="Classification of the engineering diagram"
    )
    detected_components: List[ComponentDefinition] = Field(
        default_factory=list,
        description="List of detected nodes/components with inferred pinouts and labels"
    )
    connections: List[ConnectionDefinition] = Field(
        default_factory=list,
        description="List of connections/wires between components"
    )
    design_errors: List[str] = Field(
        default_factory=list,
        description="List of warnings/errors, e.g., 'Floating node at pin 2', 'Missing ground reference', 'Dead-end step'"
    )
    suggested_fix: str = Field(
        description="Actionable engineering guidance on how to correct the drawing or complete the circuit"
    )
    mcp_tool_payload: str = Field(
        description="The exact formatted string (DOT graph definition, SPICE netlist text, or DXF coordinates JSON) ready to be sent to Stage 1 tools"
    )
    target_mcp_tool: Optional[Literal["export_to_graphviz", "generate_dxf_schematic", "validate_spice_netlist"]] = Field(
        default=None,
        description="Target Stage 1 MCP tool to invoke with mcp_tool_payload"
    )
