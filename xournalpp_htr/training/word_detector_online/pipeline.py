"""Glue between the stroke grouping detectors and the HTR pipelines."""

from functools import lru_cache

from xournalpp_htr.training.word_detector_online.data import load_pages
from xournalpp_htr.training.word_detector_online.detectors import (
    GROUPERS,
    StrokeGrouper,
)

#: IAM-OnDB forms the IAM-trained detectors are fit on.
IAM_FORMS = 700

#: Range of the factor the gaps between words are shrunk by for "iam_squeezed".
IAM_GAP_SQUEEZE = (0.3, 1.0)

#: IAM-OnDB forms the post-processing is tuned on. They come after the forms the
#: detector is fit on, so the two never share a page.
IAM_POSTPROCESS_FORMS = 300
IAM_POSTPROCESS_SKIP_FORMS = IAM_FORMS + 100


def fit_grouper(
    name: str,
    train_on: str = "benchmark",
    held_out_document: str | None = None,
    dataset_version: str | None = None,
) -> StrokeGrouper:
    """Fit detector `name` on the benchmark or on IAM-OnDB.

    ``train_on="benchmark"`` fits on the benchmark leaving out one document:
    benchmarking a pipeline on a document passes that document's file name as
    `held_out_document`, so the detector is scored on data it was not fit on
    (leave-one-document-out). ``None`` fits on the whole benchmark.

    ``train_on="hub"`` loads the published detector instead of fitting one.

    ``train_on="iam"`` fits on IAM-OnDB pages only, so the benchmark stays a
    pure test set and `held_out_document` is irrelevant. ``"iam_squeezed"`` does
    the same with the gaps between words shrunk to `IAM_GAP_SQUEEZE`, and
    ``"iam_squeezed_postprocessed"`` adds the post-processing of the groups,
    tuned on other IAM-OnDB pages.
    """
    if train_on == "hub":
        return _load_from_hub()
    if train_on == "iam":
        return _fit_on_iam(name, None, False)
    if train_on == "iam_squeezed":
        return _fit_on_iam(name, IAM_GAP_SQUEEZE, False)
    if train_on == "iam_squeezed_postprocessed":
        return _fit_on_iam(name, IAM_GAP_SQUEEZE, True)
    if train_on != "benchmark":
        raise ValueError(f"Unknown training data: {train_on}")
    return _fit_on_benchmark(name, held_out_document, dataset_version)


@lru_cache(maxsize=None)
def _fit_on_benchmark(
    name: str, held_out_document: str | None, dataset_version: str | None
) -> StrokeGrouper:
    pages = [p for p in load_pages(dataset_version) if p.document != held_out_document]
    return GROUPERS[name]().fit(pages)


@lru_cache(maxsize=None)
def _fit_on_iam(
    name: str, gap_squeeze: tuple[float, float] | None, postprocess: bool
) -> StrokeGrouper:
    # Imported here as it needs the IAM-OnDB training dependencies.
    from xournalpp_htr.training.word_detector_online.iam_ondb import (
        load_iam_ondb_pages,
    )

    pages, _ = load_iam_ondb_pages(
        IAM_FORMS, synthetic_non_text=False, gap_squeeze=gap_squeeze
    )
    grouper = GROUPERS[name]().fit(pages)
    if not postprocess:
        return grouper

    from xournalpp_htr.training.word_detector_online.postprocess import (
        PostProcessedGrouper,
        tune_postprocess,
    )

    tuning_pages, _ = load_iam_ondb_pages(
        IAM_POSTPROCESS_FORMS,
        synthetic_non_text=True,
        gap_squeeze=gap_squeeze,
        skip_forms=IAM_POSTPROCESS_SKIP_FORMS,
    )
    return PostProcessedGrouper(grouper, tune_postprocess(grouper, tuning_pages))


class _HubGrouper(StrokeGrouper):
    """The published `OnlineWordDetectorModel` as a `StrokeGrouper`."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def fit(self, pages):
        raise NotImplementedError("The published detector is loaded, not fitted.")

    def predict(self, page):
        return self.model.group_strokes(page.strokes)


@lru_cache(maxsize=None)
def _load_from_hub() -> StrokeGrouper:
    from xournalpp_htr.inference_models import OnlineWordDetectorModel

    return _HubGrouper(OnlineWordDetectorModel.from_pretrained())
