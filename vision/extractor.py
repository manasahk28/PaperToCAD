"""
Feature Extractor for Preprocessed Schematic Sketches
Stage 2: Paper-to-CAD Interactive Schematic Inspector
Extracts:
- Text and bounding boxes via PaddleOCR
- Component contours (boxes, polygons, gates) and junction dots via OpenCV
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Disable MKLDNN/oneDNN conflict on CPU for PaddlePaddle 3.x
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global cached OCR instance
_OCR_INSTANCE = None


def get_ocr_engine():
    """Lazily initializes and caches the PaddleOCR engine."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            from paddleocr import PaddleOCR
            _OCR_INSTANCE = PaddleOCR(lang="en")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            _OCR_INSTANCE = False
    return _OCR_INSTANCE if _OCR_INSTANCE is not False else None


def extract_text_and_boxes(binary_image: np.ndarray) -> Dict[str, Any]:
    """
    Runs PaddleOCR on the cleaned image to detect text labels (e.g., 'R1', '10k', 'VCC', 'GND')
    and their bounding boxes.

    Args:
        binary_image: Cleaned binary image array (from clean_sketch).

    Returns:
        Dictionary containing:
        - 'text_detections': List of detected items with text, confidence, bbox, rect [x, y, w, h], center
        - 'raw_texts': List of recognized text strings
        - 'total_detected': Count of detected text elements
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(binary_image)}")

    # PaddleOCR expects dark text on light/white background and 3-channel RGB/BGR
    if len(binary_image.shape) == 2:
        # Check if foreground is white (mean < 127 indicates black bg with white strokes)
        if np.mean(binary_image) < 127:
            ocr_input = cv2.bitwise_not(binary_image)
        else:
            ocr_input = binary_image.copy()
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)
    elif len(binary_image.shape) == 3:
        if np.mean(binary_image) < 127:
            ocr_input = cv2.bitwise_not(binary_image)
        else:
            ocr_input = binary_image.copy()
    else:
        raise ValueError(f"Invalid image dimensions: {binary_image.shape}")

    ocr = get_ocr_engine()
    text_detections: List[Dict[str, Any]] = []

    if ocr is not None:
        try:
            # Handle both PaddleOCR 3.x (predict) and 2.x (ocr)
            if hasattr(ocr, "predict"):
                pred_list = list(ocr.predict(ocr_input))
                for pred in pred_list:
                    if isinstance(pred, dict):
                        rec_texts = pred.get("rec_texts", [])
                        rec_scores = pred.get("rec_scores", [])
                        rec_polys = pred.get("rec_polys", [])
                        rec_boxes = pred.get("rec_boxes", [])

                        for idx, text_str in enumerate(rec_texts):
                            score = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0
                            if idx < len(rec_polys):
                                poly = rec_polys[idx]
                                box_pts = [[int(pt[0]), int(pt[1])] for pt in poly]
                                pts_arr = np.array(box_pts, dtype=np.int32)
                                x, y, w, h = cv2.boundingRect(pts_arr)
                            elif idx < len(rec_boxes):
                                b = rec_boxes[idx]
                                x, y, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                                w, h = x2 - x, y2 - y
                                box_pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                            else:
                                continue

                            cx = int(x + w / 2)
                            cy = int(y + h / 2)

                            text_detections.append({
                                "text": str(text_str).strip(),
                                "confidence": round(score, 4),
                                "box": box_pts,
                                "rect": [int(x), int(y), int(w), int(h)],
                                "center": [cx, cy],
                            })
            elif hasattr(ocr, "ocr"):
                results = ocr.ocr(ocr_input, cls=True)
                if results and len(results) > 0 and results[0]:
                    for line in results[0]:
                        box_coords = line[0]
                        text_str, conf = line[1]
                        pts = np.array(box_coords, dtype=np.int32)
                        x, y, w, h = cv2.boundingRect(pts)
                        cx = int(x + w / 2)
                        cy = int(y + h / 2)

                        text_detections.append({
                            "text": str(text_str).strip(),
                            "confidence": round(float(conf), 4),
                            "box": [[int(pt[0]), int(pt[1])] for pt in box_coords],
                            "rect": [int(x), int(y), int(w), int(h)],
                            "center": [cx, cy],
                        })
        except Exception as e:
            logger.warning(f"PaddleOCR execution error: {e}")

    return {
        "text_detections": text_detections,
        "raw_texts": [d["text"] for d in text_detections],
        "total_detected": len(text_detections),
    }


def detect_junctions_and_contours(
    binary_image: np.ndarray,
    min_component_area: float = 300.0,
    min_junction_radius: float = 3.5,
) -> List[Dict[str, Any]]:
    """
    Uses OpenCV contour detection, distance transforms, and geometric analysis to find:
    - Closed loops: Component body outlines (rectangles, boxes, triangles, IC blocks)
    - Junction dots: Connection nodes where wires meet or branch (using distance transform peaks,
      morphology, and circularity)

    Args:
        binary_image: Cleaned binary image with white strokes on black background (or vice versa).
        min_component_area: Minimum pixel area to be considered a component body.
        min_junction_radius: Minimum radius threshold for identifying a junction dot.

    Returns:
        List of detected dictionaries with keys:
        - type: 'box' | 'triangle' | 'polygon' | 'junction'
        - bbox: [x, y, w, h]
        - center: [cx, cy]
        - area: float
        - radius: float (for junctions)
        - vertices_count: int (for polygons)
        - contour: list of points (for polygons)
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(binary_image)}")

    # Ensure single-channel grayscale
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_image.copy()

    # Ensure white foreground (strokes) on black background
    if np.mean(gray) > 127:
        fg_binary = cv2.bitwise_not(gray)
    else:
        fg_binary = gray

    # Ensure binary values 0 and 255
    _, fg_binary = cv2.threshold(fg_binary, 127, 255, cv2.THRESH_BINARY)

    h_img, w_img = fg_binary.shape
    total_img_area = float(h_img * w_img)

    detected_items: List[Dict[str, Any]] = []
    detected_junction_centers: List[Tuple[int, int]] = []

    # 1. Distance Transform Peak Detection for Junction Dots
    # In schematics, a junction dot is a localized thickening at a wire intersection
    dist_map = cv2.distanceTransform(fg_binary, cv2.DIST_L2, 5)
    foreground_pixels = dist_map[fg_binary > 0]
    if len(foreground_pixels) > 0:
        median_stroke_radius = float(np.median(foreground_pixels))
        # Threshold for junction dots: significantly thicker than typical wire stroke
        junction_thresh = max(min_junction_radius, median_stroke_radius * 2.0)
        thick_regions = (dist_map >= junction_thresh).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thick_regions)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = float(stats[i, cv2.CC_STAT_AREA])

            # Peak distance within this connected component
            max_r = float(np.max(dist_map[labels == i]))

            # Filter out giant filled blocks or long horizontal/vertical lines
            aspect = float(w) / max(h, 1)
            if 0.4 <= aspect <= 2.5 and max_r <= 35.0:
                detected_items.append({
                    "type": "junction",
                    "bbox": [x, y, w, h],
                    "center": [int(cx), int(cy)],
                    "radius": round(max_r, 1),
                    "area": round(area, 1),
                    "detection_method": "distance_transform_peak",
                })
                detected_junction_centers.append((int(cx), int(cy)))

    # 2. Contour Detection (Component Body Outlines: boxes, triangles, polygons)
    contours, hierarchy = cv2.findContours(fg_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip noise specks and anything too small to be a component
        if area < min_component_area:
            continue

        # Skip the whole-image frame / canvas border
        if area > 0.85 * total_img_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # Check for canvas outer border
        if w >= 0.95 * w_img and h >= 0.95 * h_img:
            continue

        # Ignore very thin single wire lines (aspect ratio > 10)
        aspect_ratio = float(w) / max(h, 1)
        if aspect_ratio > 10.0 or aspect_ratio < 0.1:
            continue

        # Geometric properties
        circularity = 4.0 * math.pi * (area / (perimeter ** 2))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        # Polygon approximation
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        v_count = len(approx)

        if v_count == 4:
            shape_type = "box"
        elif v_count == 3:
            shape_type = "triangle"
        elif circularity >= 0.75:
            shape_type = "circle"
        else:
            shape_type = "polygon"

        detected_items.append({
            "type": shape_type,
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [cx, cy],
            "area": float(area),
            "vertices_count": v_count,
            "solidity": round(solidity, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "contour": cnt.reshape(-1, 2).tolist(),
        })

    return detected_items
