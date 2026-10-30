"""Glue between the stroke grouping detectors and the HTR pipelines."""

from functools import lru_cache

from xournalpp_htr.training.online_word_detector.data import load_pages
from xournalpp_htr.training.online_word_detector.detectors import (
    GROUPERS,
    StrokeGrouper,
)


@lru_cache(maxsize=None)
def fit_grouper(
    name: str,
    held_out_document: str | None = None,
    dataset_version: str | None = None,
) -> StrokeGrouper:
    """Fit detector `name` on the benchmark, leaving out one document.

    Benchmarking a pipeline on a document passes that document's file name as
    `held_out_document`, so the detector is scored on data it was not fit on
    (leave-one-document-out). ``None`` fits on the whole benchmark.
    """
    pages = [p for p in load_pages(dataset_version) if p.document != held_out_document]
    return GROUPERS[name]().fit(pages)
