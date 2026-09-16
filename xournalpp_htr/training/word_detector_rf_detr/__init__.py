"""RF-DETR-based word detector training subpackage (ADR 006 section 3).

Training code for the RF-DETR word detector model. Its dependencies are
declared as the ``training-word-detector-rf-detr`` optional extra. Inference
does *not* import this subpackage -- it uses the ONNX export via
:class:`xournalpp_htr.inference_models.RFDETRWordDetectorModel`.
"""

try:
    import rfdetr  # noqa: F401
except ImportError as e:
    raise ImportError(
        "RF-DETR word detector training requires additional dependencies. "
        "Install with: uv add xournalpp_htr[training-word-detector-rf-detr]"
    ) from e
