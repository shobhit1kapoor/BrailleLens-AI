from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from .detect import detect_dots
from .preprocess import VARIANTS, preprocess_image
from .quality import score_image_quality
from .segment import group_cells
from .translation import translate_text


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Please upload a JPG or PNG.")
    return image


def encode_png_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def build_overlay(original: np.ndarray, cells: list[dict], dots: list, debug: dict) -> np.ndarray:
    overlay = original.copy()
    for dot in dots:
        cv2.circle(overlay, (int(dot.x), int(dot.y)), max(3, int(dot.radius)), (20, 235, 150), 2)
        cv2.circle(overlay, (int(dot.x), int(dot.y)), 2, (255, 255, 255), -1)
    for y in debug.get("row_centers", []):
        cv2.line(overlay, (0, int(y)), (overlay.shape[1], int(y)), (255, 200, 40), 1)
    for x in debug.get("column_centers", []):
        cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0]), (120, 180, 255), 1)
    for cell in cells:
        x, y, w, h = cell["bbox"]
        confidence = cell["confidence"]
        color = (35, 220, 90) if confidence >= 0.78 else (50, 190, 255) if confidence >= 0.5 else (60, 60, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{cell.get('char') or '?'} {int(confidence * 100)}%"
        cv2.putText(overlay, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay


def _score_candidate(result: dict) -> float:
    text = result.get("text") or ""
    cells = result.get("cells") or []
    metrics = result.get("metrics") or {}
    quality = result.get("quality") or {}
    if not cells:
        return 0.0
    chars = [cell.get("char", "") for cell in cells]
    unknown_ratio = chars.count("?") / max(1, len(chars))
    filled_ratio = sum(1 for cell in cells if cell.get("pattern")) / max(1, len(cells))
    line_counts: dict[int, int] = {}
    for cell in cells:
        line_counts[cell.get("line", 0)] = line_counts.get(cell.get("line", 0), 0) + 1
    short_line_ratio = sum(1 for count in line_counts.values() if count < 4) / max(1, len(line_counts))
    cells_count = max(1, metrics.get("cells_detected", len(cells)))
    dots_count = metrics.get("dots_detected", 0)
    dot_per_cell = dots_count / cells_count
    line_bonus = min(0.12, (result.get("debug", {}).get("line_count", 1) or 1) * 0.025)
    text_bonus = min(0.06, len(text.replace("\n", "").strip()) / 220.0)
    dot_cell_balance = max(0.0, 1.0 - abs(dot_per_cell - 2.7) / 3.2)
    over_detection_penalty = max(0.0, (cells_count - 95) / 260.0)
    long_noise_penalty = max(0.0, (len(text.replace("\n", "")) - 140) / 360.0)
    confidence = result.get("confidence", 0.0)
    readiness = quality.get("readiness", 0.0)
    score = (
        confidence * 0.42
        + readiness * 0.14
        + filled_ratio * 0.12
        + dot_cell_balance * 0.12
        + text_bonus
        + line_bonus
        - unknown_ratio * 0.28
        - short_line_ratio * 0.12
        - over_detection_penalty * 0.22
        - long_noise_penalty * 0.18
    )
    return round(float(max(0.0, score)), 4)


def _scan_processed(image: np.ndarray, language: str, calibration: dict | None, variant: str, started_at: float) -> tuple[dict, dict, list]:
    processed = preprocess_image(image, variant)
    detection_mask = processed.get("separated_threshold", processed["threshold"])
    dots, detector_diagnostics = detect_dots(processed["gray"], detection_mask, processed.get("embossed_response"))
    cells, text, segment_debug = group_cells(dots, calibration)
    quality, warnings = score_image_quality(processed["gray"], len(dots), segment_debug.get("skew_angle", 0.0))

    if not cells:
        warnings.append("No Braille cell grid found")
    elif any(cell["confidence"] < 0.5 for cell in cells):
        warnings.append("Some cells are uncertain")

    confidence_values = [cell["confidence"] for cell in cells] or [0.0]
    overall_confidence = round(float(np.mean(confidence_values)) * quality["readiness"], 3)
    translated = translate_text(text, language)
    elapsed = int((time.perf_counter() - started_at) * 1000)

    response = {
        "text": text,
        "translated_text": translated,
        "language": language,
        "cells": cells,
        "confidence": overall_confidence,
        "warnings": list(dict.fromkeys(warnings)),
        "quality": quality,
        "metrics": {
            "processing_ms": elapsed,
            "dots_detected": len(dots),
            "cells_detected": len(cells),
        },
        "detector": detector_diagnostics,
        "debug": {
            "row_centers": segment_debug.get("row_centers", []),
            "column_centers": segment_debug.get("column_centers", []),
            "skew_angle": segment_debug.get("skew_angle", 0.0),
            "line_count": segment_debug.get("line_count", 0),
            "dot_radius": segment_debug.get("dot_radius", 0.0),
            "row_spacing": segment_debug.get("row_spacing", 0.0),
            "dot_spacing": segment_debug.get("dot_spacing", 0.0),
            "document_crop": processed.get("crop", {}),
            "selected_variant": variant,
        },
    }
    response["scanner_score"] = _score_candidate(response)
    response["detector"]["variant"] = variant
    return response, processed, dots


def scan_image(
    image_bytes: bytes,
    language: str = "en",
    calibration: dict | None = None,
    debug: bool = False,
    scan_variant: str = "auto",
) -> dict:
    start = time.perf_counter()
    image = decode_image(image_bytes)
    candidates: list[tuple[dict, dict, list]] = []
    variants = list(VARIANTS.keys()) if scan_variant in {"auto", "", None} else [scan_variant if scan_variant in VARIANTS else "balanced"]
    for variant in variants:
        try:
            candidates.append(_scan_processed(image, language, calibration, variant, start))
        except Exception:
            continue
    if not candidates:
        raise ValueError("No scanner variant could process this image.")

    candidates.sort(key=lambda item: item[0].get("scanner_score", 0.0), reverse=True)
    response, processed, dots = candidates[0]
    elapsed = int((time.perf_counter() - start) * 1000)
    response["metrics"]["processing_ms"] = elapsed
    response["metrics"]["variants_tried"] = len(candidates)
    response["metrics"]["scan_mode"] = scan_variant or "auto"
    response["metrics"]["selected_variant"] = response["debug"].get("selected_variant", "balanced")
    response["alternatives"] = [
        {
            "variant": candidate[0]["debug"].get("selected_variant"),
            "text": candidate[0].get("text"),
            "confidence": candidate[0].get("confidence"),
            "score": candidate[0].get("scanner_score"),
            "dots_detected": candidate[0].get("metrics", {}).get("dots_detected"),
            "cells_detected": candidate[0].get("metrics", {}).get("cells_detected"),
            "warnings": candidate[0].get("warnings", []),
        }
        for candidate in candidates[:4]
    ]

    if debug:
        cells = response["cells"]
        segment_debug = {
            "row_centers": response["debug"].get("row_centers", []),
            "column_centers": response["debug"].get("column_centers", []),
        }
        overlay = build_overlay(processed["original"], cells, dots, segment_debug)
        response["debug"]["overlay_image_base64"] = encode_png_base64(overlay)
        response["debug"]["threshold_image_base64"] = encode_png_base64(processed["threshold"])
        response["debug"]["separated_threshold_image_base64"] = encode_png_base64(processed["separated_threshold"])
        response["debug"]["dot_enhanced_image_base64"] = encode_png_base64(processed["dot_enhanced"])
        response["debug"]["embossed_response_base64"] = encode_png_base64(processed["embossed_response"])
    return response


def calibrate_image(image_bytes: bytes, expected_text: str = "") -> dict:
    image = decode_image(image_bytes)
    processed = preprocess_image(image)
    dots, _ = detect_dots(processed["gray"], processed.get("separated_threshold", processed["threshold"]), processed.get("embossed_response"))
    cells, _, segment_debug = group_cells(dots)
    profile = {
        "expected_text": expected_text,
        "dot_radius": segment_debug.get("dot_radius", 4.0),
        "dot_spacing": segment_debug.get("dot_spacing", 18.0),
        "row_spacing": segment_debug.get("row_spacing", 18.0),
        "cell_spacing": segment_debug.get("cell_spacing", 38.0),
        "skew_angle": segment_debug.get("skew_angle", 0.0),
        "dots_detected": len(dots),
        "cells_detected": len(cells),
    }
    return {"profile": profile, "message": "Calibration profile created for this session."}
