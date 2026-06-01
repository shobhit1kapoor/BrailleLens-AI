from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def crop_document_region(image: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(mask) < 80:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, {"document_crop": False}
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < height * width * 0.25:
        return image, {"document_crop": False}

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    ordered = order_points(box)
    (tl, tr, br, bl) = ordered
    target_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    target_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if target_width < width * 0.35 or target_height < height * 0.35:
        x, y, w, h = cv2.boundingRect(contour)
        margin = int(min(width, height) * 0.025)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + w + margin)
        y1 = min(height, y + h + margin)
        return image[y0:y1, x0:x1], {"document_crop": True, "mode": "bounding_rect", "bbox": [x0, y0, x1 - x0, y1 - y0]}
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height), borderValue=(245, 245, 245))
    return warped, {"document_crop": True, "mode": "perspective", "size": [target_width, target_height]}


def normalize_image(image: np.ndarray, max_dim: int = 1280) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


VARIANTS = {
    "balanced": {
        "clahe": 2.5,
        "background_sigma": 21,
        "flatten_alpha": 1.65,
        "flatten_beta": -0.65,
        "flatten_gamma": 112,
        "adaptive_block": 31,
        "adaptive_c": 4,
        "response_kernel": 13,
        "merge_response": False,
    },
    "printed": {
        "clahe": 2.0,
        "background_sigma": 17,
        "flatten_alpha": 1.45,
        "flatten_beta": -0.45,
        "flatten_gamma": 100,
        "adaptive_block": 29,
        "adaptive_c": 5,
        "response_kernel": 11,
        "merge_response": False,
    },
    "embossed": {
        "clahe": 3.2,
        "background_sigma": 27,
        "flatten_alpha": 1.95,
        "flatten_beta": -0.95,
        "flatten_gamma": 122,
        "adaptive_block": 35,
        "adaptive_c": 2,
        "response_kernel": 15,
        "merge_response": True,
    },
    "sensitive": {
        "clahe": 3.8,
        "background_sigma": 19,
        "flatten_alpha": 1.85,
        "flatten_beta": -0.85,
        "flatten_gamma": 118,
        "adaptive_block": 25,
        "adaptive_c": 1,
        "response_kernel": 9,
        "merge_response": True,
    },
}


def preprocess_image(image: np.ndarray, variant: str = "balanced") -> dict:
    settings = VARIANTS.get(variant, VARIANTS["balanced"])
    normalized, scale = normalize_image(image)
    normalized, crop_debug = crop_document_region(normalized)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY) if len(normalized.shape) == 3 else normalized
    clahe = cv2.createCLAHE(clipLimit=settings["clahe"], tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 7, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    background = cv2.GaussianBlur(blurred, (0, 0), sigmaX=settings["background_sigma"], sigmaY=settings["background_sigma"])
    flattened = cv2.addWeighted(blurred, settings["flatten_alpha"], background, settings["flatten_beta"], settings["flatten_gamma"])
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Raised Braille usually appears as a light/dark pair rather than a clean ink dot.
    # Combining top-hat and black-hat responses makes embossed dots stand out under uneven light.
    response_kernel_size = settings["response_kernel"]
    response_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (response_kernel_size, response_kernel_size))
    top_hat = cv2.morphologyEx(flattened, cv2.MORPH_TOPHAT, response_kernel)
    black_hat = cv2.morphologyEx(flattened, cv2.MORPH_BLACKHAT, response_kernel)
    embossed_response = cv2.addWeighted(top_hat, 0.85, black_hat, 1.15, 0)
    embossed_response = cv2.GaussianBlur(embossed_response, (3, 3), 0)
    embossed_response = cv2.normalize(embossed_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dot_enhanced = cv2.subtract(np.full_like(flattened, 255), embossed_response)
    dot_enhanced = cv2.equalizeHist(dot_enhanced)

    threshold = cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        settings["adaptive_block"],
        settings["adaptive_c"],
    )
    if settings["merge_response"]:
        _, response_threshold = cv2.threshold(embossed_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = cv2.bitwise_or(threshold, response_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Build a dot-center mask that deliberately separates touching threshold islands.
    # It is better for Braille to keep small, distinct dot centers than thick white blobs.
    response_cutoff = max(18, int(np.percentile(embossed_response, 93)))
    _, response_mask = cv2.threshold(embossed_response, response_cutoff, 255, cv2.THRESH_BINARY)
    response_mask = cv2.morphologyEx(response_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    distance = cv2.distanceTransform(response_mask, cv2.DIST_L2, 3)
    if distance.max() > 0:
        _, centers = cv2.threshold(distance, distance.max() * 0.35, 255, cv2.THRESH_BINARY)
        centers = centers.astype(np.uint8)
    else:
        centers = response_mask
    center_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    separated_threshold = cv2.dilate(centers, center_kernel, iterations=1)
    separated_threshold = cv2.morphologyEx(separated_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    separated_contours, _ = cv2.findContours(separated_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rough_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(separated_contours) < max(6, int(len(rough_contours) * 0.65)):
        separated_threshold = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return {
        "original": normalized,
        "scale": scale,
        "variant": variant,
        "crop": crop_debug,
        "gray": gray,
        "enhanced": enhanced,
        "flattened": flattened,
        "dot_enhanced": dot_enhanced,
        "embossed_response": embossed_response,
        "blurred": blurred,
        "threshold": closed,
        "separated_threshold": separated_threshold,
    }
