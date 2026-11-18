"""Video helpers: burn a text label into rendered frames (so every clip says what it shows)."""

from __future__ import annotations

import numpy as np


def label_frame(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=font)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    d.text((12, 10), text, fill=color, font=font)
    return np.asarray(img)
