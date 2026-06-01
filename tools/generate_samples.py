from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.braille_map import text_to_patterns


SAMPLE_DIR = ROOT / "sample-images"


def draw_braille(text: str = "hello world") -> np.ndarray:
    patterns = text_to_patterns(text)
    cell_w = 54
    cell_gap = 24
    dot_gap_x = 22
    dot_gap_y = 22
    margin = 56
    radius = 7
    width = margin * 2 + len(patterns) * cell_w + (len(patterns) - 1) * cell_gap
    height = margin * 2 + dot_gap_y * 2 + 32
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    rng = np.random.default_rng(42)
    paper_noise = rng.normal(0, 3, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + paper_noise, 0, 255).astype(np.uint8)

    for index, pattern in enumerate(patterns):
        x0 = margin + index * (cell_w + cell_gap)
        y0 = margin
        if pattern == "":
            continue
        for position_char in pattern:
            position = int(position_char)
            col = 0 if position <= 3 else 1
            row = (position - 1) % 3
            cx = x0 + col * dot_gap_x
            cy = y0 + row * dot_gap_y
            cv2.circle(image, (cx, cy), radius + 2, (210, 210, 210), -1)
            cv2.circle(image, (cx - 2, cy - 2), radius, (42, 42, 42), -1)
            cv2.circle(image, (cx - 4, cy - 4), 2, (95, 95, 95), -1)
    return image


def draw_braille_page(lines: list[str]) -> np.ndarray:
    line_patterns = [text_to_patterns(line) for line in lines]
    cell_w = 54
    cell_gap = 24
    dot_gap_x = 22
    dot_gap_y = 22
    line_gap = 58
    margin_x = 70
    margin_y = 64
    radius = 5
    max_cells = max(len(patterns) for patterns in line_patterns)
    width = margin_x * 2 + max_cells * cell_w + (max_cells - 1) * cell_gap
    height = margin_y * 2 + len(lines) * (dot_gap_y * 2 + 12) + (len(lines) - 1) * line_gap
    image = np.full((height, width, 3), 238, dtype=np.uint8)
    rng = np.random.default_rng(7)
    paper_noise = rng.normal(0, 2, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + paper_noise, 0, 255).astype(np.uint8)
    for line_index, patterns in enumerate(line_patterns):
        y0 = margin_y + line_index * (dot_gap_y * 2 + 12 + line_gap)
        for index, pattern in enumerate(patterns):
            x0 = margin_x + index * (cell_w + cell_gap)
            if pattern == "":
                continue
            for position_char in pattern:
                position = int(position_char)
                col = 0 if position <= 3 else 1
                row = (position - 1) % 3
                cx = x0 + col * dot_gap_x
                cy = y0 + row * dot_gap_y
                cv2.circle(image, (cx + 2, cy + 2), radius + 2, (186, 186, 186), -1)
                cv2.circle(image, (cx - 2, cy - 2), radius + 1, (252, 252, 252), -1)
                cv2.circle(image, (cx, cy), radius, (45, 45, 45), -1)
                cv2.circle(image, (cx - 2, cy - 2), 2, (105, 105, 105), -1)

    shadow = np.linspace(1.02, 0.88, height, dtype=np.float32)[:, None, None]
    image = np.clip(image.astype(np.float32) * shadow, 0, 255).astype(np.uint8)
    return image


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(246, 246, 246))


def main() -> None:
    SAMPLE_DIR.mkdir(exist_ok=True)
    clear = draw_braille()
    cv2.imwrite(str(SAMPLE_DIR / "hello_world_clear.png"), clear)
    cv2.imwrite(str(SAMPLE_DIR / "hello_world_tilted.png"), rotate(clear, -6))
    low_light = np.clip(clear.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SAMPLE_DIR / "hello_world_low_light.png"), low_light)
    blurry = cv2.GaussianBlur(clear, (11, 11), 0)
    cv2.imwrite(str(SAMPLE_DIR / "hello_world_blurry.png"), blurry)
    page = draw_braille_page(["hello world", "braille reader", "scan text"])
    cv2.imwrite(str(SAMPLE_DIR / "realistic_page_embossed.png"), rotate(page, -1.0))


if __name__ == "__main__":
    main()
