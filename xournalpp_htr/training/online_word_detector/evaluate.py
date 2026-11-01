"""Leave-one-document-out cross-validation of the stroke grouping detectors.

Run with ``python -m xournalpp_htr.training.online_word_detector.evaluate``.
The benchmark has only a handful of documents, each by a different writer, so
each document is a fold and holding one out also holds out its writer; folds
with few words are listed but left out of the macro average.
"""

import argparse
from collections import defaultdict

from xournalpp_htr.training.online_word_detector.data import StrokePage, load_pages
from xournalpp_htr.training.online_word_detector.detectors import GROUPERS
from xournalpp_htr.training.online_word_detector.metrics import (
    GroupCounts,
    box_counts,
    group_counts,
)

MIN_WORDS_FOR_MACRO = 10


METRICS = {"exact": group_counts, "box": box_counts}


def cross_validate(
    name: str,
    pages: list[StrokePage],
    beta: float,
    extra_train: list[StrokePage] = (),
    use_benchmark_train: bool = True,
) -> dict[str, dict[str, GroupCounts]]:
    """Return the held-out `GroupCounts` of detector `name`, per metric and document.

    ``extra_train`` pages (e.g. IAM-OnDB) are added to every training set;
    ``use_benchmark_train=False`` trains on them alone.

    ``exact`` requires a predicted group to be exactly a ground truth word;
    ``box`` uses the benchmark's box-coverage matching.
    """
    by_document: dict[str, list[StrokePage]] = defaultdict(list)
    for page in pages:
        by_document[page.document].append(page)

    results: dict[str, dict[str, GroupCounts]] = {m: {} for m in METRICS}
    shared_grouper = None
    if not use_benchmark_train:
        # The training data does not depend on the held-out document.
        shared_grouper = GROUPERS[name](beta=beta).fit(list(extra_train))
    for held_out, test_pages in by_document.items():
        if shared_grouper is not None:
            grouper = shared_grouper
        else:
            train_pages = list(extra_train) + [
                p for p in pages if p.document != held_out
            ]
            grouper = GROUPERS[name](beta=beta).fit(train_pages)
        predictions = [grouper.predict(page) for page in test_pages]
        for metric, count in METRICS.items():
            results[metric][held_out] = sum(
                (
                    count(pred, page)
                    for pred, page in zip(predictions, test_pages, strict=True)
                ),
                GroupCounts(),
            )
    return results


def print_table(results: dict[str, GroupCounts]) -> None:
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
    parser.add_argument(
        "--train-on",
        choices=["benchmark", "iam", "both"],
        default="benchmark",
        help=(
            "Training data: the other benchmark documents (leave-one-document-out), "
            "IAM-OnDB lines only, or both"
        ),
    )
    parser.add_argument("--iam-forms", type=int, default=700)
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add synthetic non-text strokes to the IAM-OnDB pages",
    )
    args = parser.parse_args()

    pages = load_pages(args.dataset_version)
    print(f"{len(pages)} pages, {len({p.document for p in pages})} documents")

    iam_pages = []
    if args.train_on != "benchmark":
        from xournalpp_htr.training.online_word_detector.iam_ondb import (
            load_iam_ondb_pages,
        )

        iam_pages, stats = load_iam_ondb_pages(
            args.iam_forms, synthetic_non_text=args.synthetic
        )
        print(f"IAM-OnDB pages: {dict(stats)}")

    for name in args.detectors:
        results = cross_validate(
            name,
            pages,
            args.beta,
            extra_train=iam_pages,
            use_benchmark_train=args.train_on != "iam",
        )
        for metric in METRICS:
            print(
                f"\n== {name}, {metric} match, train on {args.train_on} "
                f"(tuned for F{args.beta:g}) =="
            )
            print_table(results[metric])


if __name__ == "__main__":
    main()
