"""Bootstrap confidence intervals for benchmark runs, from crop-analysis folders.

Reads the ``manifest.csv`` of two or more crop-analysis folders (written by
``scripts/run_benchmark.py --crop-analysis``) and resamples the ground truth
words, once word by word and once document by document, to show how much of a
difference between pipelines could be chance on this small benchmark.

The metrics follow ``xournalpp_htr.benchmark``: recall is matched words over
ground truth words, CER pools the edit distance (case-insensitive) and the
characters over matched words, and the combined score is recall * (1 - CER).
Words of two pipelines are paired by document, page and text, in order of
appearance.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


@dataclass
class Words:
    """The ground truth words of one pipeline run, one entry per word."""

    keys: list[tuple]  # (document, page, text, occurrence)
    documents: np.ndarray  # document index per word
    matched: np.ndarray
    edits: np.ndarray  # case-insensitive edit distance, 0 if not matched
    chars: np.ndarray  # characters of matched words, 0 if not matched
    n_spurious: int


def load_words(folder: Path) -> Words:
    rows = list(csv.DictReader(open(folder / "manifest.csv")))
    seen = defaultdict(int)
    keys, matched, edits, chars, documents = [], [], [], [], []
    document_ids = {}
    n_spurious = 0
    for row in rows:
        if row["status"] == "spurious":
            n_spurious += 1
            continue
        key_base = (row["sample"], row["page"], row["gt_text"])
        keys.append((*key_base, seen[key_base]))
        seen[key_base] += 1
        documents.append(document_ids.setdefault(row["sample"], len(document_ids)))
        is_matched = row["status"] != "missed"
        matched.append(is_matched)
        gt = row["gt_text"].lower()
        edits.append(edit_distance(gt, row["pred_text"].lower()) if is_matched else 0)
        chars.append(len(gt) if is_matched else 0)
    return Words(
        keys,
        np.array(documents),
        np.array(matched, dtype=float),
        np.array(edits, dtype=float),
        np.array(chars, dtype=float),
        n_spurious,
    )


def metrics(w: Words, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics for resampled words; `weights` has shape (resamples, words)."""
    recall = (weights * w.matched).sum(axis=1) / weights.sum(axis=1)
    cer = (weights * w.edits).sum(axis=1) / np.maximum(
        (weights * w.chars).sum(axis=1), 1.0
    )
    return {"recall": recall, "CER": cer, "R x (1-CER)": recall * (1 - cer)}


def resample_weights(
    documents: np.ndarray, n: int, by_document: bool, rng: np.random.Generator
) -> np.ndarray:
    n_words = len(documents)
    if not by_document:
        return rng.multinomial(n_words, np.full(n_words, 1 / n_words), size=n).astype(
            float
        )
    n_documents = documents.max() + 1
    doc_counts = rng.multinomial(
        n_documents, np.full(n_documents, 1 / n_documents), size=n
    )
    return doc_counts[:, documents].astype(float)


def interval(samples: np.ndarray) -> str:
    low, high = np.percentile(samples, [2.5, 97.5])
    return f"[{low:.3f}, {high:.3f}]"


def pair(a: Words, b: Words) -> tuple[Words, Words]:
    """Restrict both runs to the words found in both, in the same order."""
    index_b = {key: i for i, key in enumerate(b.keys)}
    take_a = [i for i, key in enumerate(a.keys) if key in index_b]
    take_b = [index_b[a.keys[i]] for i in take_a]

    def subset(w: Words, take: list[int]) -> Words:
        return Words(
            [w.keys[i] for i in take],
            w.documents[take],
            w.matched[take],
            w.edits[take],
            w.chars[take],
            w.n_spurious,
        )

    return subset(a, take_a), subset(b, take_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = {folder.name: load_words(folder) for folder in args.folders}
    rng = np.random.default_rng(args.seed)

    for by_document in (False, True):
        level = "document" if by_document else "word"
        print(f"\n== 95% intervals, resampling {level}s ==")
        for name, w in runs.items():
            weights = resample_weights(w.documents, args.resamples, by_document, rng)
            point = metrics(w, np.ones((1, len(w.matched))))
            print(f"{name}  ({len(w.matched)} words, {int(w.matched.sum())} matched)")
            for metric, samples in metrics(w, weights).items():
                print(f"  {metric:<12} {point[metric][0]:.3f}  {interval(samples)}")

    names = list(runs)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            a, b = pair(runs[names[first]], runs[names[second]])
            print(
                f"\n== {names[first]} minus {names[second]} "
                f"({len(a.matched)} paired words) =="
            )
            for by_document in (False, True):
                level = "document" if by_document else "word"
                weights = resample_weights(
                    a.documents, args.resamples, by_document, rng
                )
                ma, mb = metrics(a, weights), metrics(b, weights)
                pa = metrics(a, np.ones((1, len(a.matched))))
                pb = metrics(b, np.ones((1, len(b.matched))))
                for metric in ma:
                    diff = ma[metric] - mb[metric]
                    print(
                        f"  by {level:<8} {metric:<12} "
                        f"{pa[metric][0] - pb[metric][0]:+.3f}  {interval(diff)}  "
                        f"first better in {np.mean(diff > 0):.0%} of resamples"
                    )


if __name__ == "__main__":
    main()
