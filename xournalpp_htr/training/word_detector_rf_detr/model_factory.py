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


def build_model(variant: str, resolution: int, pretrain_weights: str | None = None):
    """Instantiate an RF-DETR model, optionally from a trained checkpoint."""
    model_cls = resolve_model_class(variant)
    kwargs: dict = {"resolution": resolution}
    if pretrain_weights is not None:
        kwargs["pretrain_weights"] = pretrain_weights
    return model_cls(**kwargs)
