from __future__ import annotations

import cv2
import numpy as np


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def score_image_quality(gray: np.ndarray, dots_count: int = 0, skew_angle: float = 0.0) -> tuple[dict, list[str]]:
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_raw = float(np.mean(gray))
    contrast_raw = float(np.std(gray))

    focus = clamp01(laplacian_var / 180.0)
    if 80 <= brightness_raw <= 245:
        brightness = 1.0
    elif brightness_raw < 80:
        brightness = clamp01(brightness_raw / 80.0)
    else:
        brightness = clamp01(1.0 - (brightness_raw - 245.0) / 10.0)
    contrast = clamp01(contrast_raw / 70.0)
    alignment = clamp01(1.0 - abs(skew_angle) / 18.0)
    dot_grid = clamp01(dots_count / 24.0)
    readiness = clamp01((focus * 0.25) + (brightness * 0.2) + (contrast * 0.2) + (alignment * 0.15) + (dot_grid * 0.2))

    warnings: list[str] = []
    if focus < 0.42:
        warnings.append("Too blurry")
    if brightness_raw < 75:
        warnings.append("Lighting is low")
    elif brightness_raw > 248 and contrast < 0.35:
        warnings.append("Lighting is too harsh")
    if contrast < 0.35:
        warnings.append("Low contrast may reduce dot detection")
    if abs(skew_angle) > 8:
        warnings.append("Slight tilt detected")
    if dots_count and dots_count < 8:
        warnings.append("Dot grid is weak")
    if readiness > 0.75 and dots_count >= 8:
        warnings.append("Ready to scan")
    elif dots_count >= 8:
        warnings.append("Braille detected")

    return {
        "focus": round(focus, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "alignment": round(alignment, 3),
        "dot_grid": round(dot_grid, 3),
        "readiness": round(readiness, 3),
        "raw": {
            "laplacian_variance": round(laplacian_var, 2),
            "mean_luminance": round(brightness_raw, 2),
            "contrast_stddev": round(contrast_raw, 2),
            "skew_angle": round(skew_angle, 2),
        },
    }, warnings
