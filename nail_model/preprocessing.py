"""
AnemiaSense — Nail Preprocessing Pipeline
Implements the 10-step verified input pipeline for the MobileNetV3Large model.

CRITICAL ARCHITECTURAL CONTRACT:
The model contains an internal Rescaling(scale=1/127.5, offset=-1.0) layer that
maps [0, 255] to [-1.0, 1.0] inside the computational graph.
Input must be supplied as raw float32 pixels in the range [0.0, 255.0].
External normalization to [-1, 1] must NOT be applied to avoid double normalization.
"""

from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


def validate_nail_image(image_input: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
    """
    Validates that the input is a readable, non-empty RGB/BGR image.
    Rejects corrupted, empty, or unreadable inputs with descriptive ValueError.
    Returns RGB uint8 numpy array with shape (H, W, 3).
    """
    if image_input is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Input image bytes are empty (0 bytes).")
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image bytes. Unsupported or corrupted file format.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif hasattr(image_input, "convert") and hasattr(image_input, "mode"):
        # PIL.Image.Image instance
        img_rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise ValueError(f"Image file not found: {image_input}")
        with open(p, "rb") as f:
            raw_bytes = f.read()
        return validate_nail_image(raw_bytes)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input image array is empty (size 0).")
        if image_input.ndim == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                # Assume already RGB if passed from web handler
                img_rgb = image_input.copy()
            else:
                raise ValueError(f"Unsupported channel count: {image_input.shape[2]}")
        else:
            raise ValueError(f"Unsupported array dimensions: {image_input.ndim}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img_rgb.shape[0] < 16 or img_rgb.shape[1] < 16:
        raise ValueError(f"Image resolution too small: {img_rgb.shape[1]}x{img_rgb.shape[0]}. Minimum is 16x16.")

    return img_rgb


def evaluate_nail_image_quality(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates optical quality of the nail image (blur, brightness, glare).
    Returns quality metrics and warnings.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    glare_pct = float(np.mean(gray > 245) * 100.0)

    blur_min = 20.0
    min_lum = 20.0
    max_lum = 235.0
    glare_max_pct = 15.0

    passed = (blur_score >= blur_min and min_lum <= brightness <= max_lum and glare_pct <= glare_max_pct)
    warnings = []
    if blur_score < blur_min:
        warnings.append("Nail image appears out of focus or blurred. Hold camera steady.")
    if brightness < min_lum:
        warnings.append("Illumination is low. Adequate ambient lighting is recommended.")
    elif brightness > max_lum:
        warnings.append("Image is overexposed. Avoid direct harsh flashlight.")
    if glare_pct > glare_max_pct:
        warnings.append("Specular glare/reflection detected on nail plate. Angle camera away from direct glare.")

    return {
        "blur_score": round(blur_score, 2),
        "blur_threshold": blur_min,
        "brightness": round(brightness, 2),
        "glare_percentage": round(glare_pct, 2),
        "passed": bool(passed),
        "warnings": warnings,
    }


def preprocess_nail_image(
    image_input: Union[np.ndarray, bytes, bytearray],
    target_size: Tuple[int, int] = (224, 224),
    crop_coords: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Full 10-step verified preprocessing pipeline for MobileNetV3Large nail model:
      1. Accept uploaded image
      2. Validate that it is a readable image
      3. Reject unsupported or corrupted inputs with clear error
      4. Convert image to RGB
      5. Apply optional user crop coordinates (if provided)
      6. Aspect-ratio preserving square padding
      7. Resize to target_size (224x224)
      8. Convert into correct numerical format: float32 in [0.0, 255.0]
      9. Add batch dimension -> shape (1, 224, 224, 3)
      10. Validate final tensor shape and value range

    Returns:
      tensor: np.ndarray of shape (1, 224, 224, 3), dtype float32, range [0, 255]
      padded_rgb: np.ndarray of shape (target_size[0], target_size[1], 3), uint8 (for Grad-CAM overlay)
      qa_metrics: Dict containing optical quality assessment
    """
    # Steps 1 - 4: Accept, validate, reject bad input, convert to RGB
    img_rgb = validate_nail_image(image_input)
    h_orig, w_orig = img_rgb.shape[:2]

    # Step 5: Region of Interest (ROI) / Interactive Crop
    crop = img_rgb
    if crop_coords:
        try:
            parts = [float(p.strip()) for p in crop_coords.split(",") if p.strip()]
            if len(parts) == 4:
                cx, cy, cw, ch = parts
                if cw <= 1.0 and ch <= 1.0:
                    x0 = max(0, int(cx * w_orig))
                    y0 = max(0, int(cy * h_orig))
                    w0 = min(w_orig - x0, int(cw * w_orig))
                    h0 = min(h_orig - y0, int(ch * h_orig))
                else:
                    x0 = max(0, int(cx))
                    y0 = max(0, int(cy))
                    w0 = min(w_orig - x0, int(cw))
                    h0 = min(h_orig - y0, int(ch))
                if w0 > 15 and h0 > 15:
                    crop = img_rgb[y0 : y0 + h0, x0 : x0 + w0]
        except Exception as e:
            print(f"[AnemiaSense Nail] Crop parsing warning: {e}")

    # Step 6: Aspect-ratio preserving square padding
    h_c, w_c = crop.shape[:2]
    max_dim = max(h_c, w_c)
    padded_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_off = (max_dim - h_c) // 2
    x_off = (max_dim - w_c) // 2
    padded_canvas[y_off : y_off + h_c, x_off : x_off + w_c] = crop

    # Step 7: Resize to target size (224x224)
    padded_resized = cv2.resize(padded_canvas, target_size, interpolation=cv2.INTER_AREA)

    # Step 8: Convert to numerical format: float32 in [0.0, 255.0]
    # Internal Rescaling layer in model takes [0, 255] and handles [-1, 1] internally
    float_tensor = padded_resized.astype(np.float32)

    # Step 9: Add batch dimension -> (1, 224, 224, 3)
    batch_tensor = np.expand_dims(float_tensor, axis=0)

    # Step 10: Validate final tensor shape and values
    expected_shape = (1, target_size[0], target_size[1], 3)
    if batch_tensor.shape != expected_shape:
        raise ValueError(f"Preprocessing tensor shape error: got {batch_tensor.shape}, expected {expected_shape}")

    if not np.isfinite(batch_tensor).all():
        raise ValueError("Preprocessing tensor contains non-finite values (NaN or Inf).")

    min_val, max_val = float(batch_tensor.min()), float(batch_tensor.max())
    if min_val < 0.0 or max_val > 255.0:
        raise ValueError(f"Preprocessing pixel range out of bounds: [{min_val}, {max_val}]. Expected [0, 255].")

    qa_metrics = evaluate_nail_image_quality(crop)

    return batch_tensor, padded_resized, qa_metrics
