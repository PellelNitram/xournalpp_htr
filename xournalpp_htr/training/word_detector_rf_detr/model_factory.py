"""Resolve ``model.variant`` strings to ``rfdetr`` model classes.

``rfdetr`` exposes its model classes through a module-level ``__getattr__``
(lazy loading), so ``dir()`` does not necessarily list them and the package
does not define ``__version__``. Everything in here is written against that
behaviour so that train/export/predict give the same, useful error when a
variant name is wrong.
"""

import importlib.metadata

import rfdetr


def rfdetr_version() -> str:
    """Installed ``rfdetr`` version, or ``"unknown"``.

    ``rfdetr`` has no ``__version__`` attribute, so ask the package metadata.
    """
    try:
        return importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def available_variants() -> list[str]:
    """Variant names the installed ``rfdetr`` exposes.

    Prefers ``__all__`` because lazily-loaded classes need not appear in
    ``dir()``.
    """
    names = getattr(rfdetr, "__all__", None) or dir(rfdetr)
    return sorted(
        name.removeprefix("RFDETR").lower()
        for name in names
        if name.startswith("RFDETR") and name != "RFDETR"
    )


def resolve_model_class(variant: str):
    """Return the ``rfdetr.RFDETR<Variant>`` class for ``variant``."""
    class_name = f"RFDETR{variant.capitalize()}"
    try:
        return getattr(rfdetr, class_name)
    except AttributeError:
        available = ", ".join(available_variants()) or "none found"
        raise ValueError(
            f"Unknown variant {variant!r} ({class_name} not found in "
            f"rfdetr {rfdetr_version()}). Available: {available}."
        ) from None


def validate_resolution(model, resolution: int) -> None:
    """Raise if ``resolution`` is not valid for this model variant.

    RF-DETR requires the input resolution to be a positive multiple of
    ``patch_size * num_windows``. That product is read off the model rather
    than hardcoded, because it varies by variant.
    """
    mc = model.model_config
    patch_size = getattr(mc, "patch_size", None)
    num_windows = getattr(mc, "num_windows", None)
    if patch_size is None or num_windows is None:
        return  # unknown layout; let rfdetr do its own validation
    divisor = patch_size * num_windows
    if resolution <= 0 or resolution % divisor != 0:
        nearest = max(divisor, round(resolution / divisor) * divisor)
        raise ValueError(
            f"resolution={resolution} is invalid for this variant: it must be "
            f"a positive multiple of patch_size * num_windows = "
            f"{patch_size} * {num_windows} = {divisor}. Nearest valid value: "
            f"{nearest}."
        )


def build_model(variant: str, resolution: int, pretrain_weights: str | None = None):
    """Instantiate an RF-DETR model, optionally from a trained checkpoint."""
    model_cls = resolve_model_class(variant)
    kwargs: dict = {"resolution": resolution}
    if pretrain_weights is not None:
        kwargs["pretrain_weights"] = pretrain_weights
    model = model_cls(**kwargs)
    validate_resolution(model, resolution)
    return model
