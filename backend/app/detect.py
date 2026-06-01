from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np


@dataclass
class DotCandidate:
    x: float
    y: float
    radius: float
    area: float
    confidence: float
    method: str


def _merge_candidates(candidates: list[DotCandidate], distance: float = 7.0) -> list[DotCandidate]:
    merged: list[DotCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        match = next(
            (
                current
                for current in merged
                if ((current.x - candidate.x) ** 2 + (current.y - candidate.y) ** 2) ** 0.5 < max(distance, candidate.radius)
            ),
            None,
        )
        if match is None:
            merged.append(candidate)
        elif candidate.confidence > match.confidence:
            match.x = candidate.x
            match.y = candidate.y
            match.radius = candidate.radius
            match.area = candidate.area
            match.confidence = candidate.confidence
            match.method = f"{match.method}+{candidate.method}"
    return sorted(merged, key=lambda item: (item.y, item.x))


def _filter_grid_consistency(candidates: list[DotCandidate]) -> list[DotCandidate]:
    if len(candidates) < 18:
        return candidates
    median_radius = float(np.median([candidate.radius for candidate in candidates]))
    max_neighbor_distance = max(18.0, median_radius * 8.0)
    min_neighbor_distance = max(3.0, median_radius * 1.1)
    filtered: list[DotCandidate] = []
    for candidate in candidates:
        if candidate.confidence >= 0.88:
            filtered.append(candidate)
            continue
        neighbor_count = 0
        for other in candidates:
            if other is candidate:
                continue
            dx = abs(other.x - candidate.x)
            dy = abs(other.y - candidate.y)
            distance = (dx * dx + dy * dy) ** 0.5
            if min_neighbor_distance <= distance <= max_neighbor_distance:
                neighbor_count += 1
        if neighbor_count >= 1:
            filtered.append(candidate)
    return filtered or candidates


def detect_contours(binary: np.ndarray) -> list[DotCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    image_area = height * width
    candidates: list[DotCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 6 or area > image_area * 0.02:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 1.5 or radius > min(width, height) * 0.08:
            continue
        confidence = min(1.0, 0.35 + circularity * 0.45 + min(area / 90.0, 0.2))
        candidates.append(DotCandidate(x, y, radius, area, confidence, "contour"))
    return candidates


def detect_blobs(binary: np.ndarray) -> list[DotCandidate]:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 6
    params.maxArea = max(80, binary.shape[0] * binary.shape[1] * 0.02)
    params.filterByCircularity = True
    params.minCircularity = 0.35
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 10
    params.maxThreshold = 240
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary)
    return [
        DotCandidate(kp.pt[0], kp.pt[1], kp.size / 2, pi * (kp.size / 2) ** 2, 0.72, "blob")
        for kp in keypoints
    ]


def detect_hough(gray: np.ndarray) -> list[DotCandidate]:
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=8,
        param1=55,
        param2=13,
        minRadius=2,
        maxRadius=max(6, min(gray.shape[:2]) // 35),
    )
    if circles is None:
        return []
    candidates: list[DotCandidate] = []
    for x, y, radius in np.round(circles[0, :]).astype("int"):
        candidates.append(DotCandidate(float(x), float(y), float(radius), float(pi * radius * radius), 0.58, "hough"))
    return candidates


def detect_response_peaks(response: np.ndarray) -> list[DotCandidate]:
    percentile = float(np.percentile(response, 91))
    threshold_value = max(18.0, percentile)
    _, mask = cv2.threshold(response, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = response.shape[:2]
    image_area = height * width
    candidates: list[DotCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5 or area > image_area * 0.01:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * pi * area / (perimeter * perimeter)
        if circularity < 0.25:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if radius < 1.2 or radius > min(width, height) * 0.035:
            continue
        local_score = float(response[int(round(y)), int(round(x))]) / 255.0
        confidence = min(1.0, 0.35 + local_score * 0.42 + circularity * 0.25)
        candidates.append(DotCandidate(x, y, radius, area, confidence, "embossed"))
    return candidates


def detect_dots(gray: np.ndarray, binary: np.ndarray, response: np.ndarray | None = None) -> tuple[list[DotCandidate], dict]:
    contour_candidates = detect_contours(binary)
    blob_candidates = detect_blobs(binary)
    response_candidates = detect_response_peaks(response) if response is not None else []
    hough_candidates = detect_hough(gray) if len(contour_candidates) + len(blob_candidates) + len(response_candidates) < 8 else []
    merged = _filter_grid_consistency(_merge_candidates(contour_candidates + blob_candidates + response_candidates + hough_candidates))
    diagnostics = {
        "contour_candidates": len(contour_candidates),
        "blob_candidates": len(blob_candidates),
        "embossed_candidates": len(response_candidates),
        "hough_candidates": len(hough_candidates),
        "merged_candidates": len(merged),
        "method": "hybrid-contour-blob-embossed-hough",
    }
    return merged, diagnostics
