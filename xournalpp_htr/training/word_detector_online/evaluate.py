"""Score the stroke grouping detector on the benchmark.

Run with ``python -m xournalpp_htr.training.word_detector_online.evaluate``.
The detector is trained on IAM-OnDB pages (see ``iam_ondb.py``), so no benchmark
document is involved in training and every document is a fresh test. Documents
with few words are listed but left out of the macro average.
"""

import argparse
from collections import defaultdict

from xournalpp_htr.training.word_detector_online.data import StrokePage, load_pages
from xournalpp_htr.training.word_detector_online.detectors import (
    GROUPERS,
    StrokeGrouper,
)
from xournalpp_htr.training.word_detector_online.iam_ondb import (
    TRAINING_FORMS,
    TRAINING_GAP_SQUEEZE,
    load_iam_ondb_pages,
)
from xournalpp_htr.training.word_detector_online.metrics import (
    GroupCounts,
    box_counts,
    group_counts,
)

MIN_WORDS_FOR_MACRO = 10

METRICS = {"exact": group_counts, "box": box_counts}


def score(
    grouper: StrokeGrouper, pages: list[StrokePage]
) -> dict[str, dict[str, GroupCounts]]:
    """Return the `GroupCounts` of a fitted detector, per metric and document.

    ``exact`` requires a predicted group to be exactly a ground truth word;
    ``box`` uses the benchmark's box-coverage matching.
    """
    by_document: dict[str, list[StrokePage]] = defaultdict(list)
    for page in pages:
        by_document[page.document].append(page)

    results: dict[str, dict[str, GroupCounts]] = {m: {} for m in METRICS}
    for document, document_pages in by_document.items():
        predictions = [grouper.predict(page) for page in document_pages]
        for metric, count in METRICS.items():
            results[metric][document] = sum(
                (
                    count(prediction, page)
                    for prediction, page in zip(
                        predictions, document_pages, strict=True
                    )
                ),
                GroupCounts(),
            )
    return results


def print_table(results: dict[str, GroupCounts]) -> None:
    print(f"{'document':<28}{'words':>6}{'P':>7}{'R':>7}{'F1':>7}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="beta of the F-beta score the detector's merge threshold is tuned for",
    )
    parser.add_argument("--detectors", nargs="+", default=["pairwise_gbm_tuned"])
    parser.add_argument("--iam-forms", type=int, default=TRAINING_FORMS)
    parser.add_argument(
        "--iam-seed",
        type=int,
        default=0,
        help="Seed of the random draw of IAM-OnDB forms",
    )
    parser.add_argument(
        "--gap-squeeze",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=TRAINING_GAP_SQUEEZE,
        help=(
            "Shrink the gaps between words of the IAM-OnDB lines by a factor "
            "drawn uniformly from [LOW, HIGH] per line; 1 1 leaves them as they are"
        ),
    )
    args = parser.parse_args()

    pages = load_pages(args.dataset_version)
    print(f"{len(pages)} benchmark pages, {len({p.document for p in pages})} documents")

    train_pages, stats = load_iam_ondb_pages(
        args.iam_forms, seed=args.iam_seed, gap_squeeze=tuple(args.gap_squeeze)
    )
    print(f"IAM-OnDB pages: {dict(stats)}")

    for name in args.detectors:
        grouper = GROUPERS[name](beta=args.beta).fit(train_pages)
        results = score(grouper, pages)
        for metric in METRICS:
            print(f"\n== {name}, {metric} match (tuned for F{args.beta:g}) ==")
            print_table(results[metric])


if __name__ == "__main__":
    main()
