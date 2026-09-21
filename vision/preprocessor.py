"""
Computer Vision Preprocessor for Hand-Drawn Sketches
Stage 2: Paper-to-CAD Interactive Schematic Inspector
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def clean_sketch(
    image_path: Union[str, Path, np.ndarray],
    invert: bool = False,
    return_black_ink: bool = False,
) -> np.ndarray:
    """
    Cleans hand-drawn smartphone sketches:
    - Converts to grayscale.
    - Removes lighting gradients, shadows, and non-uniform illumination using morphological
      background estimation and bilateral filtering.
    - Applies adaptive thresholding (Otsu's binarization) to create a crisp binary mask of strokes.

    Args:
        image_path: Filepath string, Path object, or already-loaded numpy image array (BGR or Gray).
        invert: If True, returns inverted mask.
        return_black_ink: If True, returns black ink on white background (ideal for standard OCR).
                          By default (False), returns white strokes (255) on black background (0),
                          which is the standard format for contour detection in OpenCV.

    Returns:
        Binary uint8 numpy array with pixel values in {0, 255}.
    """
    # 1. Load image
    if isinstance(image_path, (str, Path)):
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Failed to decode image from path: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError(f"Unsupported image_path type: {type(image_path)}")

    # 2. Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3. Remove lighting gradients and shadows
    # Use morphological closing/dilation with a large structuring element to estimate background illumination
    h, w = gray.shape
    kernel_size = max(15, min(h, w) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1

    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Dilate to extract the background illumination level
    bg_illumination = cv2.morphologyEx(gray, cv2.MORPH_DILATE, bg_kernel)

    # Avoid zero-division and normalize division: gray / bg_illumination * 255
    bg_illumination = np.maximum(bg_illumination, 1)
    normalized = cv2.divide(gray, bg_illumination, scale=255.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Apply bilateral filter to smooth paper grain/texture while keeping stroke edges sharp
    filtered = cv2.bilateralFilter(normalized, d=7, sigmaColor=50, sigmaSpace=50)

    # 4. Adaptive Thresholding / Otsu's Binarization
    # Compute Otsu's threshold on inverted image (ink as foreground)
    # Background in normalized image is close to 255, ink strokes are dark (< 150)
    _, binary_otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Optional morphological cleanup of tiny 1-pixel noise specks
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, clean_kernel)

    # Return mode
    if return_black_ink:
        return cv2.bitwise_not(cleaned)
    elif invert:
        return cv2.bitwise_not(cleaned)
    else:
        return cleaned
