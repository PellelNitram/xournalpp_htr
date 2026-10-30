"""Leave-one-document-out cross-validation of the stroke grouping detectors.

Run with ``python -m xournalpp_htr.training.online_word_detector.evaluate``.
The benchmark has only a handful of documents by a single writer, so each
document is a fold; folds with few words are listed but left out of the macro
average.
"""

import argparse
from collections import defaultdict

from xournalpp_htr.training.online_word_detector.data import StrokePage, load_pages
from xournalpp_htr.training.online_word_detector.detectors import GROUPERS
from xournalpp_htr.training.online_word_detector.metrics import (
    GroupCounts,
    group_counts,
)

MIN_WORDS_FOR_MACRO = 10


def cross_validate(
    name: str, pages: list[StrokePage], beta: float
) -> dict[str, GroupCounts]:
    """Return the held-out `GroupCounts` of detector `name` per document."""
    by_document: dict[str, list[StrokePage]] = defaultdict(list)
    for page in pages:
        by_document[page.document].append(page)

    results = {}
    for held_out, test_pages in by_document.items():
        train_pages = [p for p in pages if p.document != held_out]
        grouper = GROUPERS[name](beta=beta).fit(train_pages)
        counts = GroupCounts()
        for page in test_pages:
            counts = counts + group_counts(grouper.predict(page), page)
        results[held_out] = counts
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="beta of the F-beta score the detectors are tuned for",
    )
    parser.add_argument("--detectors", nargs="+", default=list(GROUPERS))
    args = parser.parse_args()

    pages = load_pages(args.dataset_version)
    print(f"{len(pages)} pages, {len({p.document for p in pages})} documents")

    for name in args.detectors:
        results = cross_validate(name, pages, args.beta)
        print(f"\n== {name} (tuned for F{args.beta:g}) ==")
        print(f"{'held-out document':<28}{'words':>6}{'P':>7}{'R':>7}{'F1':>7}")
        for document, c in results.items():
            print(
                f"{document[:26]:<28}{c.n_gt:>6}{c.precision:>7.2f}"
                f"{c.recall:>7.2f}{c.f_beta():>7.2f}"
            )
        pooled = sum(results.values(), GroupCounts())
        macro = [c for c in results.values() if c.n_gt >= MIN_WORDS_FOR_MACRO]
        print(
            f"{'pooled':<28}{pooled.n_gt:>6}{pooled.precision:>7.2f}"
            f"{pooled.recall:>7.2f}{pooled.f_beta():>7.2f}"
        )
        print(
            f"{'macro (>=%d words)' % MIN_WORDS_FOR_MACRO:<28}{'':>6}"
            f"{sum(c.precision for c in macro) / len(macro):>7.2f}"
            f"{sum(c.recall for c in macro) / len(macro):>7.2f}"
            f"{sum(c.f_beta() for c in macro) / len(macro):>7.2f}"
        )


if __name__ == "__main__":
    main()
