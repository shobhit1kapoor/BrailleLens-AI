from __future__ import annotations

from statistics import median
from math import cos, radians, sin

import cv2
import numpy as np

from .braille_map import CAPITAL_SIGN, NUMBER_SIGN, PATTERN_TO_CHAR, PUNCTUATION, translate_patterns
from .detect import DotCandidate


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - median(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(median(cluster)) for cluster in clusters]


def _median_spacing(values: list[float], fallback: float) -> float:
    diffs = [b - a for a, b in zip(values, values[1:]) if b - a > 2]
    if not diffs:
        return fallback
    small_diffs = [diff for diff in diffs if diff <= np.percentile(diffs, 45)]
    return float(median(small_diffs or diffs))


def estimate_skew(dots: list[DotCandidate]) -> float:
    if len(dots) < 4:
        return 0.0
    points = np.float32([[dot.x, dot.y] for dot in dots])
    rect = cv2.minAreaRect(points)
    angle = float(rect[-1])
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    return angle


def _nearest_index(value: float, centers: list[float]) -> int:
    return min(range(len(centers)), key=lambda index: abs(centers[index] - value))


def _estimate_cell_pitch(column_centers: list[float], dot_spacing: float) -> float:
    diffs = [b - a for a, b in zip(column_centers, column_centers[1:]) if b - a > 2]
    big_diffs = [diff for diff in diffs if diff > dot_spacing * 1.45]
    if big_diffs:
        regular_gap = float(np.percentile(big_diffs, 25))
        pitch = regular_gap + dot_spacing
        return float(max(dot_spacing * 2.65, min(dot_spacing * 3.85, pitch)))
    return dot_spacing * 3.2


def _line_row_groups(row_centers: list[float], row_spacing: float) -> list[list[float]]:
    if len(row_centers) <= 3:
        return [row_centers[:3]]
    groups: list[list[float]] = []
    current: list[float] = []
    for row in row_centers:
        if current and row - current[-1] > row_spacing * 1.85:
            if len(current) >= 2:
                groups.append(current[:3])
            current = []
        current.append(row)
        if len(current) == 3:
            groups.append(current)
            current = []
    if len(current) >= 2:
        groups.append(current[:3])
    return groups


def _score_anchors(line_dots: list[DotCandidate], anchors: list[float], dot_spacing: float) -> int:
    score = 0
    tolerance = max(4.0, dot_spacing * 0.55)
    for dot in line_dots:
        anchor = anchors[_nearest_index(dot.x, anchors)]
        if min(abs(dot.x - anchor), abs(dot.x - (anchor + dot_spacing))) <= tolerance:
            score += 1
    return score


def _build_line_anchors(line_dots: list[DotCandidate], dot_spacing: float, calibration: dict | None) -> list[float]:
    xs = sorted(dot.x for dot in line_dots)
    if not xs:
        return []
    column_centers = _cluster_values(xs, max(4.0, dot_spacing * 0.42))
    line_dot_spacing = float((calibration or {}).get("dot_spacing", _median_spacing(column_centers, dot_spacing)))
    cell_pitch = float(
        (calibration or {}).get(
            "cell_pitch",
            (calibration or {}).get("cell_spacing", _estimate_cell_pitch(column_centers, line_dot_spacing)),
        )
    )
    min_x = min(xs)
    max_x = max(xs)
    candidates = [min_x, min_x - line_dot_spacing]
    best_anchors: list[float] = []
    best_score = -1
    for start in candidates:
        count = int(round((max_x - start) / cell_pitch)) + 1
        anchors = [start + i * cell_pitch for i in range(max(1, count + 1)) if start + i * cell_pitch <= max_x + cell_pitch * 0.55]
        score = _score_anchors(line_dots, anchors, line_dot_spacing)
        if score > best_score:
            best_score = score
            best_anchors = anchors
    return best_anchors


def _is_known_pattern(pattern: str) -> bool:
    return pattern in PATTERN_TO_CHAR or pattern in PUNCTUATION or pattern in {NUMBER_SIGN, CAPITAL_SIGN}


def _should_keep_cell(pattern: str, confidence: float, dot_count: int) -> bool:
    if not pattern:
        return False
    if _is_known_pattern(pattern):
        return dot_count >= 1 and confidence >= 0.3
    return dot_count >= 4 and confidence >= 0.74


def group_cells(dots: list[DotCandidate], calibration: dict | None = None) -> tuple[list[dict], str, dict]:
    if not dots:
        return [], "", {"row_centers": [], "column_centers": [], "skew_angle": 0.0}

    skew_angle = estimate_skew(dots)
    if abs(skew_angle) > 1.0:
        cx = float(median([dot.x for dot in dots]))
        cy = float(median([dot.y for dot in dots]))
        theta = radians(-skew_angle)
        rotated: list[DotCandidate] = []
        for dot in dots:
            dx = dot.x - cx
            dy = dot.y - cy
            rotated.append(
                DotCandidate(
                    x=cx + dx * cos(theta) - dy * sin(theta),
                    y=cy + dx * sin(theta) + dy * cos(theta),
                    radius=dot.radius,
                    area=dot.area,
                    confidence=dot.confidence,
                    method=dot.method,
                )
            )
        dots = rotated

    xs = sorted(dot.x for dot in dots)
    ys = sorted(dot.y for dot in dots)
    dot_radius = float(median([dot.radius for dot in dots])) if dots else 4.0
    row_spacing_hint = float((calibration or {}).get("row_spacing", dot_radius * 4.4))
    col_spacing_hint = float((calibration or {}).get("dot_spacing", dot_radius * 4.0))
    row_centers = _cluster_values(ys, max(5.0, row_spacing_hint * 0.42))
    column_centers = _cluster_values(xs, max(5.0, col_spacing_hint * 0.42))

    row_spacing = float((calibration or {}).get("row_spacing", _median_spacing(row_centers, dot_radius * 4.4)))
    dot_spacing = float((calibration or {}).get("dot_spacing", _median_spacing(column_centers, dot_radius * 4.0)))

    if len(row_centers) >= 3:
        normalized_rows = []
        current = row_centers[0]
        for row in row_centers:
            if not normalized_rows or row - normalized_rows[-1] > row_spacing * 0.55:
                normalized_rows.append(row if len(normalized_rows) == 0 else row)
            current = row
        row_centers = normalized_rows

    dot_lookup: dict[tuple[int, int], DotCandidate] = {}
    cells: list[dict] = []
    all_row_centers: list[float] = []
    all_column_centers: list[float] = []
    line_texts: list[str] = []
    index = 0
    row_groups = _line_row_groups(row_centers, row_spacing)
    from .braille_map import pattern_to_char

    for line_number, line_rows in enumerate(row_groups):
        if len(line_rows) < 2:
            continue
        while len(line_rows) < 3:
            line_rows.append(line_rows[-1] + row_spacing)
        y_min = line_rows[0] - row_spacing * 0.8
        y_max = line_rows[2] + row_spacing * 0.8
        line_dots = [dot for dot in dots if y_min <= dot.y <= y_max]
        if len(line_dots) < 2:
            continue
        cell_anchors = _build_line_anchors(line_dots, dot_spacing, calibration)
        if not cell_anchors:
            continue
        raw_cells: list[dict] = []
        line_lookup: dict[tuple[int, int], DotCandidate] = {}
        line_column_centers: list[float] = []
        for anchor in cell_anchors:
            line_column_centers.extend([anchor, anchor + dot_spacing])

        for dot in line_dots:
            row_index = _nearest_index(dot.y, line_rows)
            cell_index = _nearest_index(dot.x, cell_anchors)
            local_col = 0 if abs(dot.x - cell_anchors[cell_index]) <= abs(dot.x - (cell_anchors[cell_index] + dot_spacing)) else 1
            col_index = cell_index * 2 + local_col
            if abs(dot.y - line_rows[row_index]) > row_spacing * 0.65 or abs(dot.x - line_column_centers[col_index]) > dot_spacing * 0.68:
                continue
            current = line_lookup.get((row_index, col_index))
            if current is None or dot.confidence > current.confidence:
                line_lookup[(row_index, col_index)] = dot

        for cell_index, anchor in enumerate(cell_anchors):
            start_col = cell_index * 2

            cell_dots: list[dict] = []
            pattern_positions: list[str] = []
            confidences: list[float] = []
            for local_col in range(2):
                for local_row in range(3):
                    row_index = local_row
                    col_index = start_col + local_col
                    if col_index >= len(line_column_centers):
                        continue
                    position = local_row + 1 if local_col == 0 else local_row + 4
                    found = line_lookup.get((row_index, col_index))
                    if found:
                        pattern_positions.append(str(position))
                        confidences.append(found.confidence)
                        cell_dots.append(
                            {
                                "position": position,
                                "x": round(found.x, 1),
                                "y": round(found.y, 1),
                                "confidence": round(found.confidence, 3),
                            }
                        )

            pattern = "".join(sorted(pattern_positions))
            x0 = anchor - dot_spacing * 0.55
            y0 = line_rows[0] - row_spacing * 0.55
            bbox = [int(x0), int(y0), int(dot_spacing * 2.1), int(row_spacing * 3.1)]
            confidence = float(median(confidences)) if confidences else 0.0
            raw_cells.append(
                {
                    "anchor": anchor,
                    "line": line_number,
                    "pattern": pattern,
                    "char": "",
                    "confidence": round(confidence, 3),
                    "bbox": bbox,
                    "dots": cell_dots,
                }
            )

        kept_cells = [
            cell
            for cell in raw_cells
            if _should_keep_cell(cell["pattern"], cell["confidence"], len(cell["dots"]))
        ]
        if len(kept_cells) < 2:
            continue

        expected_pitch = float(
            (calibration or {}).get(
                "cell_pitch",
                (calibration or {}).get("cell_spacing", _estimate_cell_pitch(column_centers, dot_spacing)),
            )
        )
        line_patterns: list[str] = []
        previous_anchor: float | None = None
        for cell in kept_cells:
            anchor = float(cell.pop("anchor"))
            if previous_anchor is not None and anchor - previous_anchor > expected_pitch * 1.55:
                line_patterns.append("")
            line_patterns.append(cell["pattern"])
            previous_anchor = anchor

            cell["index"] = index
            index += 1

        line_text = translate_patterns(line_patterns)
        if line_text:
            line_texts.append(line_text)
        number_mode = False
        capitalize = False
        for cell in kept_cells:
            pattern = cell["pattern"]
            char, number_mode, capitalize = pattern_to_char(pattern, number_mode, capitalize)
            cell["char"] = char or ""
            if char == "?":
                cell["confidence"] = min(cell["confidence"], 0.35)
        cells.extend(kept_cells)
        all_row_centers.extend(line_rows)
        for cell in kept_cells:
            x, _, w, _ = cell["bbox"]
            all_column_centers.extend([x + dot_spacing * 0.55, x + dot_spacing * 0.55 + dot_spacing])

    text = "\n".join(line_texts).strip()

    debug = {
        "row_centers": [round(value, 1) for value in all_row_centers or row_centers],
        "column_centers": [round(value, 1) for value in all_column_centers or column_centers],
        "skew_angle": round(skew_angle, 2),
        "dot_radius": round(dot_radius, 2),
        "row_spacing": round(row_spacing, 2),
        "dot_spacing": round(dot_spacing, 2),
        "cell_spacing": round(_estimate_cell_pitch(column_centers, dot_spacing), 2),
        "cell_pitch": round(_estimate_cell_pitch(column_centers, dot_spacing), 2),
        "line_count": len(line_texts),
    }
    return cells, text, debug
