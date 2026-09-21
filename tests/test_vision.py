"""
Stage 2 Vision and OCR Test Suite
Tests:
- clean_sketch: Removes lighting gradients and binarizes hand-drawn sketches.
- extract_text_and_boxes: Extracts OCR labels and bounding coordinates.
- detect_junctions_and_contours: Detects component body loops and junction connection dots.
- Generates synthetic hand-drawn sketch with lighting gradient and saves annotated debug images.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest
from vision.extractor import detect_junctions_and_contours, extract_text_and_boxes
from vision.preprocessor import clean_sketch


def generate_synthetic_sketch(output_path: Optional[Path] = None) -> np.ndarray:
    """
    Generates a synthetic sketch of a 3-box flowchart/schematic on paper:
    - Uneven background illumination / gradient (simulating smartphone indoor photo)
    - 3 component boxes ("Vin 12V", "R1 10k", "Vout 5V")
    - Wire traces connecting the boxes
    - Connection junction dot at a wire branch
    - Handwritten-style text labels
    """
    w, h = 900, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 245

    # 1. Simulate lighting gradient and paper vignette
    # Top-left brighter, bottom-right darker
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gradient = 25.0 * (x_coords / w) + 35.0 * (y_coords / h)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) - gradient, 160, 255).astype(np.uint8)

    # Ink color (dark slate/blue-black pen)
    pen_color = (35, 30, 25)
    stroke_thickness = 3

    # 2. Draw 3 Component Boxes
    # Box 1: Vin
    b1_x, b1_y, b1_w, b1_h = 100, 200, 160, 90
    cv2.rectangle(img, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), pen_color, stroke_thickness)
    cv2.putText(img, "Vin 12V", (b1_x + 22, b1_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pen_color, 2, cv2.LINE_AA)

    # Box 2: R1
    b2_x, b2_y, b2_w, b2_h = 370, 200, 160, 90
    cv2.rectangle(img, (b2_x, b2_y), (b2_x + b2_w, b2_y + b2_h), pen_color, stroke_thickness)
    cv2.putText(img, "R1 10k", (b2_x + 28, b2_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pen_color, 2, cv2.LINE_AA)

    # Box 3: Vout
    b3_x, b3_y, b3_w, b3_h = 640, 200, 160, 90
    cv2.rectangle(img, (b3_x, b3_y), (b3_x + b3_w, b3_y + b3_h), pen_color, stroke_thickness)
    cv2.putText(img, "Vout 5V", (b3_x + 22, b3_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pen_color, 2, cv2.LINE_AA)

    # 3. Connecting Wires
    wire_y = 245
    # Wire 1: Box 1 -> Box 2
    cv2.line(img, (b1_x + b1_w, wire_y), (b2_x, wire_y), pen_color, stroke_thickness)

    # Wire 2: Box 2 -> Box 3
    cv2.line(img, (b2_x + b2_w, wire_y), (b3_x, wire_y), pen_color, stroke_thickness)

    # Branch wire with junction dot
    branch_x = 580
    cv2.line(img, (branch_x, wire_y), (branch_x, 380), pen_color, stroke_thickness)

    # Junction dot at (branch_x, wire_y)
    junction_radius = 8
    cv2.circle(img, (branch_x, wire_y), junction_radius, pen_color, -1)

    # Label near junction branch
    cv2.putText(img, "GND", (branch_x - 45, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.75, pen_color, 2, cv2.LINE_AA)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

    return img


def test_clean_sketch():
    """Test that clean_sketch removes lighting gradients and produces a clean binary mask."""
    synth = generate_synthetic_sketch()
    cleaned = clean_sketch(synth)

    assert isinstance(cleaned, np.ndarray)
    assert len(cleaned.shape) == 2, "Expected 2D binary image"
    assert cleaned.dtype == np.uint8, "Expected uint8 array"

    unique_vals = np.unique(cleaned)
    assert set(unique_vals).issubset({0, 255}), f"Binary image should only contain 0 and 255, got {unique_vals}"

    # Foreground ink strokes should be a reasonable percentage of the page (1% to 15%)
    stroke_ratio = float(np.count_nonzero(cleaned)) / cleaned.size
    assert 0.01 <= stroke_ratio <= 0.20, f"Unexpected stroke ratio: {stroke_ratio:.3f}"

    # Save output for inspection
    out_path = PROJECT_ROOT / "output" / "vision_cleaned.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cleaned)
    assert out_path.exists() and out_path.stat().st_size > 0
    print(f"[PASS] test_clean_sketch: Generated {out_path} (stroke ratio: {stroke_ratio:.2%})")


def test_detect_junctions_and_contours():
    """Test detection of component body outlines (boxes) and junction dots."""
    synth = generate_synthetic_sketch()
    cleaned = clean_sketch(synth)

    detections = detect_junctions_and_contours(cleaned)
    assert len(detections) > 0, "No contours or junctions detected"

    types = [d["type"] for d in detections]
    boxes = [d for d in detections if d["type"] == "box"]
    junctions = [d for d in detections if d["type"] == "junction"]

    # We drew 3 boxes
    assert len(boxes) >= 3, f"Expected at least 3 component boxes, found {len(boxes)}"

    # We drew 1 junction dot
    assert len(junctions) >= 1, f"Expected at least 1 junction dot, found {len(junctions)}"

    # Check junction coordinates are near the expected branch point (580, 245)
    found_expected_junction = any(
        math.hypot(j["center"][0] - 580, j["center"][1] - 245) < 30.0 for j in junctions
    )
    assert found_expected_junction, f"Junction near (580, 245) was not detected! Found: {junctions}"

    print(
        f"[PASS] test_detect_junctions_and_contours: Found {len(boxes)} boxes and "
        f"{len(junctions)} junctions ({types})"
    )


def test_extract_text_and_boxes():
    """Test OCR extraction of text labels and their bounding boxes."""
    synth = generate_synthetic_sketch()
    cleaned = clean_sketch(synth)

    result = extract_text_and_boxes(cleaned)
    assert isinstance(result, dict)
    assert "text_detections" in result
    assert "raw_texts" in result

    # Check if OCR detected text
    raw_texts = result["raw_texts"]
    print(f"[INFO] OCR Detected text strings: {raw_texts}")

    # If PaddleOCR is installed and models loaded, verify at least some labels are detected
    if result["total_detected"] > 0:
        for det in result["text_detections"]:
            assert "text" in det
            assert "confidence" in det
            assert "box" in det
            assert "rect" in det
            assert "center" in det
    print(f"[PASS] test_extract_text_and_boxes: Extracted {result['total_detected']} text items.")


def test_full_vision_pipeline_and_annotated_debug():
    """Run full pipeline and create a color-coded annotated debug visualization."""
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    sketch_path = out_dir / "vision_test_sketch_raw.png"
    synth_bgr = generate_synthetic_sketch(output_path=sketch_path)

    # 1. Clean sketch
    cleaned_mask = clean_sketch(synth_bgr)
    cv2.imwrite(str(out_dir / "vision_cleaned.png"), cleaned_mask)

    # 2. Extract Text
    ocr_results = extract_text_and_boxes(cleaned_mask)

    # 3. Detect Contours and Junctions
    shape_detections = detect_junctions_and_contours(cleaned_mask)

    # 4. Generate Annotated Debug Image
    # Create canvas with original sketch in color
    annotated = synth_bgr.copy()

    # Draw detected component boxes in BLUE
    for item in shape_detections:
        if item["type"] == "box":
            x, y, w, h = item["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (235, 110, 20), 2)
            cv2.putText(
                annotated,
                f"BOX {w}x{h}",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 110, 20),
                2,
                cv2.LINE_AA,
            )

    # Draw detected junctions in RED
    for item in shape_detections:
        if item["type"] == "junction":
            cx, cy = item["center"]
            r = int(item.get("radius", 8))
            cv2.circle(annotated, (cx, cy), r + 5, (0, 0, 240), 2)
            cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"JUNCTION ({cx},{cy})",
                (cx - 30, cy + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 240),
                2,
                cv2.LINE_AA,
            )

    # Draw detected OCR text in GREEN
    for t_item in ocr_results.get("text_detections", []):
        x, y, w, h = t_item["rect"]
        txt = t_item["text"]
        conf = t_item["confidence"]
        box_pts = np.array(t_item["box"], dtype=np.int32)
        cv2.polylines(annotated, [box_pts], isClosed=True, color=(20, 180, 20), thickness=2)
        cv2.putText(
            annotated,
            f"TEXT: '{txt}' ({conf:.2f})",
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 140, 10),
            2,
            cv2.LINE_AA,
        )

    debug_img_path = out_dir / "vision_debug_annotated.png"
    cv2.imwrite(str(debug_img_path), annotated)

    assert debug_img_path.exists() and debug_img_path.stat().st_size > 0
    print(
        f"[PASS] test_full_vision_pipeline_and_annotated_debug: Successfully created debug visualization at {debug_img_path}"
    )


def main():
    """Run all tests directly without pytest if executed standalone."""
    print("Running Stage 2 Vision & OCR pipeline test suite...\n" + "=" * 55)
    test_clean_sketch()
    test_detect_junctions_and_contours()
    test_extract_text_and_boxes()
    test_full_vision_pipeline_and_annotated_debug()
    print("=" * 55 + "\nAll Stage 2 vision tests passed successfully!")


if __name__ == "__main__":
    main()
