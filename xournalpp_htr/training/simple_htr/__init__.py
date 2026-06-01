"""SimpleHTR training subpackage (ADR 006 section 3).

Training code for the SimpleHTR text recognition model. Its dependencies are
declared as the ``training-simple-htr`` optional extra. Inference does *not*
import this subpackage -- it uses the ONNX export via
:class:`xournalpp_htr.inference_models.SimpleHTRModel`.
"""

try:
    import tensorboard  # noqa: F401
except ImportError as e:
    raise ImportError(
        "SimpleHTR training requires additional dependencies. "
        "Install with: uv add xournalpp_htr[training-simple-htr]"
    ) from e
