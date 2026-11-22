"""Ink features for OnlineHTR. Numpy only, shared by training and inference.

Follows Carbune et al. (2020), minus the time channel: Xournal++ stores no
timestamps, so the model sees ``(dx, dy, n)`` only.

1. Shift to the ink's top-left and scale so the ink height is 1.
2. Resample every stroke at equal *arc-length* spacing (``points_per_unit``
   points per unit height). The reference interpolated in normalised time,
   which only makes sense with a real time channel.
3. Take differences between consecutive points. The jump from the end of one
   stroke to the start of the next is kept, it carries the spacing of letters
   and words.
4. ``n`` is 1 on the first point of a stroke, else 0.

``y`` follows the document convention (pointing down).
"""

import numpy as np

Stroke = tuple[np.ndarray, np.ndarray]

#: Number of feature channels.
NUM_CHANNELS = 3

_EPS = 1e-8


def _normalise(strokes: list[Stroke]) -> list[Stroke]:
    x_all = np.concatenate([x for x, _ in strokes])
    y_all = np.concatenate([y for _, y in strokes])
    x0, y0 = x_all.min(), y_all.min()
    scale = y_all.max() - y0
    if scale < _EPS:  # a flat line: fall back to its width
        scale = x_all.max() - x0
    if scale < _EPS:  # a single dot
        scale = 1.0
    return [((x - x0) / scale, (y - y0) / scale) for x, y in strokes]


def _resample(x: np.ndarray, y: np.ndarray, points_per_unit: float) -> Stroke:
    """Resample one stroke at equal arc-length spacing, keeping both ends."""
    if len(x) < 2:
        return x, y
    steps = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    length = arc[-1]
    if length < _EPS:  # all points coincide, i.e. a dot
        return x[:1], y[:1]
    n_points = max(2, int(np.ceil(length * points_per_unit)) + 1)
    grid = np.linspace(0.0, length, n_points)
    return np.interp(grid, arc, x), np.interp(grid, arc, y)


def strokes_to_features(
    strokes: list[Stroke], points_per_unit: float = 20.0
) -> np.ndarray:
    """Turn strokes into the ``(T, 3)`` float32 network input ``(dx, dy, n)``.

    :param strokes: ``(x, y)`` arrays per stroke, in writing order, y down.
    :param points_per_unit: Resampling density per unit of ink height.
    :returns: Features; an empty ``(0, 3)`` array if there are no points.
    """
    strokes = [(np.asarray(x, float), np.asarray(y, float)) for x, y in strokes]
    strokes = [(x, y) for x, y in strokes if len(x) > 0]
    if not strokes:
        return np.zeros((0, NUM_CHANNELS), dtype=np.float32)

    resampled = [_resample(x, y, points_per_unit) for x, y in _normalise(strokes)]
    x = np.concatenate([s[0] for s in resampled])
    y = np.concatenate([s[1] for s in resampled])
    starts = np.zeros(len(x))
    starts[np.cumsum([0] + [len(s[0]) for s in resampled[:-1]])] = 1.0

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return np.stack([dx, dy, starts], axis=1).astype(np.float32)
