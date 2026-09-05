"""YOLO-based word detector training subpackage (ADR 006 section 3).

Training code for the YOLO word detector model. Its dependencies are declared
as the ``training-word-detector-yolo`` optional extra. Inference does *not*
import this subpackage -- it uses the ONNX export via
:class:`xournalpp_htr.inference_models.YOLOWordDetectorModel`.
"""

try:
    import ultralytics  # noqa: F401
except ImportError as e:
    raise ImportError(
        "YOLO word detector training requires additional dependencies. "
        "Install with: uv add xournalpp_htr[training-word-detector-yolo]"
    ) from e
