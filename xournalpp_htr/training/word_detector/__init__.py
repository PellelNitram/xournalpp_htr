"""WordDetectorNN training subpackage (ADR 006 section 3).

Training code for the WordDetector model. Its dependencies are declared as the
``training-word-detector`` optional extra. Inference does *not* import this
subpackage -- it uses the ONNX export via
:class:`xournalpp_htr.inference_models.WordDetectorModel`.
"""

try:
    import tensorboard  # noqa: F401
except ImportError as e:
    raise ImportError(
        "WordDetector training requires additional dependencies. "
        "Install with: uv add xournalpp_htr[training-word-detector]"
    ) from e
