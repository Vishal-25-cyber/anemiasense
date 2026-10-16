"""
AnemiaSense — Image Preprocessing and Quality Assurance Pipeline
Implements optical quality checks, verified conjunctiva ROI extraction,
background standardization, aspect-ratio preserving padding, and MobileNetV2 normalization.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Union
import cv2
import numpy as np


class ImageQualityChecker:
    """Evaluates optical quality criteria: blur, illumination, and specular glare."""

    def __init__(self, blur_min: float = 20.0, min_lum: float = 20.0, max_lum: float = 235.0, glare_max_pct: float = 15.0):
        self.blur_min = blur_min
        self.min_lum = min_lum
        self.max_lum = max_lum
        self.glare_max_pct = glare_max_pct

    def evaluate(self, img_rgb: np.ndarray) -> Dict[str, Union[float, bool]]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        glare_pct = float(np.mean(gray > 245) * 100.0)

        passed = (blur_score >= self.blur_min) and (self.min_lum <= brightness <= self.max_lum) and (glare_pct <= self.glare_max_pct)
        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "glare_percentage": glare_pct,
            "passed": bool(passed)
        }


class ConjunctivaPreprocessor:
    """Reproducible preprocessing pipeline tailored for conjunctiva images."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), margin_pixels: int = 5):
        self.target_size = target_size
        self.margin_pixels = margin_pixels
        self.quality_checker = ImageQualityChecker()

    def process_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Reads image, detects background color, isolates conjunctiva tissue,
        standardizes background to neutral black, pads to square, resizes to target_size,
        and normalizes to [-1.0, 1.0] for MobileNetV2.
        """
        im = cv2.imread(str(image_path))
        if im is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Detect background color by sampling image corners
        corners = np.vstack([
            im[:10, :10].reshape(-1, 3), im[:10, -10:].reshape(-1, 3),
            im[-10:, :10].reshape(-1, 3), im[-10:, -10:].reshape(-1, 3)
        ])
        is_white_bg = np.all(corners.mean(axis=0) > 220)

        # Generate tissue mask
        mask = (im < 235).any(axis=-1) if is_white_bg else (im > 20).any(axis=-1)
        ys, xs = np.where(mask)

        if len(ys) == 0:
            crop = im_rgb
        else:
            y_min = max(0, ys.min() - self.margin_pixels)
            y_max = min(im.shape[0], ys.max() + self.margin_pixels)
            x_min = max(0, xs.min() - self.margin_pixels)
            x_max = min(im.shape[1], xs.max() + self.margin_pixels)

            # Standardize background to neutral black
            masked = np.zeros_like(im_rgb)
            masked[mask] = im_rgb[mask]
            crop = masked[y_min:y_max, x_min:x_max]

        # Aspect-ratio preserving square padding
        h, w, _ = crop.shape
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = crop

        resized = cv2.resize(padded, self.target_size, interpolation=cv2.INTER_AREA)
        # MobileNetV2 scaling to [-1.0, 1.0]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return normalized

    def augment_sample(self, img_uint8: np.ndarray) -> List[np.ndarray]:
        """
        Generates physiologically safe augmentations (flips, slight rotation, minor brightness).
        Explicitly excludes hue shifts to preserve biological erythema and tissue pallor.
        """
        augmented = []
        # 1. Original normalized
        augmented.append((img_uint8.astype(np.float32) / 127.5) - 1.0)

        # 2. Horizontal Flip + slight brightness jitter
        flip = cv2.flip(img_uint8, 1)
        bright = np.clip(flip.astype(np.float32) * np.random.uniform(0.96, 1.04), 0, 255).astype(np.uint8)
        augmented.append((bright.astype(np.float32) / 127.5) - 1.0)

        # 3. Small rotation (+/- 5 deg)
        angle = np.random.uniform(-5.0, 5.0)
        h, w = img_uint8.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale=np.random.uniform(0.97, 1.03))
        rotated = cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        augmented.append((rotated.astype(np.float32) / 127.5) - 1.0)

        return augmented
