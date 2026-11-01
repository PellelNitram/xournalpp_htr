"""Synthetic non-text strokes to add to pages made of text only.

IAM-OnDB has no arrows, underlines or drawings, so a detector trained on it
alone never learns to leave such strokes out of a word. The generators here
draw the kinds of non-text strokes seen in the benchmark relative to a line of
text, in the units of that line.
"""

import numpy as np

Stroke = tuple[np.ndarray, np.ndarray]

#: Kinds of non-text elements `random_element` draws.
KINDS = ("underline", "separator", "arrow", "box", "connector")


def _line_stroke(
    start: tuple[float, float],
    end: tuple[float, float],
    unit: float,
    rng: np.random.Generator,
) -> Stroke:
    """A hand-drawn straight stroke: evenly spaced points with smooth wobble."""
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    n = max(8, int(length / unit * 6))
    t = np.linspace(0.0, 1.0, n)
    wobble = np.convolve(rng.normal(0.0, 0.05 * unit, n), np.ones(5) / 5, mode="same")
    normal = np.array([-(end[1] - start[1]), end[0] - start[0]]) / max(length, 1e-9)
    x = start[0] + t * (end[0] - start[0]) + wobble * normal[0]
    y = start[1] + t * (end[1] - start[1]) + wobble * normal[1]
    return x, y


def _span(
    xmin: float, xmax: float, rng: np.random.Generator, min_share: float = 0.15
) -> tuple[float, float]:
    """A random horizontal part of the line, as a word or a few words span."""
    width = xmax - xmin
    length = rng.uniform(min_share, 0.6) * width
    start = rng.uniform(xmin, xmax - length)
    return start, start + length


def random_element(
    kind: str,
    line_box: tuple[float, float, float, float],
    unit: float,
    rng: np.random.Generator,
) -> list[Stroke]:
    """Draw one non-text element next to a text line.

    :param kind: One of `KINDS`.
    :param line_box: ``xmin, ymin, xmax, ymax`` of the line, y pointing down.
    :param unit: Typical stroke size of the line, the length all sizes scale with.
    :returns: The strokes of the element in drawing order.
    """
    xmin, ymin, xmax, ymax = line_box

    if kind == "underline":
        x0, x1 = _span(xmin, xmax, rng)
        y = ymax + rng.uniform(0.1, 0.4) * unit
        return [_line_stroke((x0, y), (x1, y + rng.normal(0, 0.1) * unit), unit, rng)]

    if kind == "separator":
        y = ymax + rng.uniform(0.8, 1.5) * unit
        left = xmin - rng.uniform(0.0, 2.0) * unit
        right = xmax + rng.uniform(0.0, 2.0) * unit
        return [_line_stroke((left, y), (right, y), unit, rng)]

    if kind == "arrow":
        start = (rng.uniform(xmin, xmax), ymax + rng.uniform(0.5, 1.5) * unit)
        angle = rng.choice([0.0, np.pi, np.pi / 2, -np.pi / 2]) + rng.normal(0, 0.3)
        length = rng.uniform(3.0, 8.0) * unit
        tip = (start[0] + length * np.cos(angle), start[1] + length * np.sin(angle))
        head = 0.6 * unit
        back = angle + np.pi
        left = (
            tip[0] + head * np.cos(back + 0.5),
            tip[1] + head * np.sin(back + 0.5),
        )
        right = (
            tip[0] + head * np.cos(back - 0.5),
            tip[1] + head * np.sin(back - 0.5),
        )
        return [
            _line_stroke(start, tip, unit, rng),
            (
                np.array([left[0], tip[0], right[0]]),
                np.array([left[1], tip[1], right[1]]),
            ),
        ]

    if kind == "box":
        x0, x1 = _span(xmin, xmax, rng)
        pad = 0.3 * unit
        corners = [
            (x0 - pad, ymin - pad),
            (x1 + pad, ymin - pad),
            (x1 + pad, ymax + pad),
            (x0 - pad, ymax + pad),
            (x0 - pad, ymin - pad),
        ]
        sides = [
            _line_stroke(a, b, unit, rng)
            for a, b in zip(corners, corners[1:], strict=False)
        ]
        return [
            (
                np.concatenate([s[0] for s in sides]),
                np.concatenate([s[1] for s in sides]),
            )
        ]

    if kind == "connector":
        # A long swooping curve across the text, like the ones seen in notes.
        n = 80
        t = np.linspace(0.0, 1.0, n)
        y_start = ymin + rng.uniform(0.0, 1.0) * (ymax - ymin)
        drift = rng.uniform(-3.0, 3.0) * unit
        smooth = t * t * (3 - 2 * t)
        wave = (
            rng.uniform(0.3, 1.5) * unit * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t)
        )
        x = xmin + t * (xmax - xmin)
        y = y_start + drift * smooth + wave
        return [(x, y)]

    raise ValueError(f"Unknown non-text element kind: {kind}")
