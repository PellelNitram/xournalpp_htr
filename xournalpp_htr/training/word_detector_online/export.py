"""Export the trained WordDetectorOnline detector as the inference artifact.

The detector trains in minutes, so there is no separate checkpoint: this script
trains it on IAM-OnDB pages (700 forms, gaps between words shrunk) and writes the files consumed by
:class:`xournalpp_htr.inference_models.OnlineWordDetectorModel`:

* ``model.npz``   -- the boosted trees as plain arrays (no pickle, no ONNX).
* ``config.json`` -- the merge threshold, feature names and training setup.

Usage::

    uv run --extra training-simple-htr-ondb python -m \\
        xournalpp_htr.training.word_detector_online.export \\
        --output-dir experiments/word_detector_online/export

The IAM-OnDB loader needs the dependencies of the ``training-simple-htr-ondb``
extra and access to the private IAM-OnDB dataset repository.
"""

import argparse
import json
import subprocess
from pathlib import Path

from xournalpp_htr.training.word_detector_online.detectors import (
    GROUPERS,
    PairwiseClassifierGrouper,
)
from xournalpp_htr.training.word_detector_online.features import PAIR_FEATURE_NAMES
from xournalpp_htr.training.word_detector_online.iam_ondb import (
    TRAINING_FORMS,
    TRAINING_GAP_SQUEEZE,
    load_iam_ondb_pages,
)
from xournalpp_htr.training.word_detector_online.model import TreeEnsemble

HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-online"
DETECTOR = "pairwise_gbm_tuned"
MODEL_CARD = Path(__file__).parent / "model_card.md"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_artifacts(
    grouper: PairwiseClassifierGrouper, output_dir: Path, training: dict
) -> dict[str, Path]:
    """Write ``model.npz`` and ``config.json`` of a fitted pairwise grouper."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.npz"
    TreeEnsemble.from_sklearn(grouper._model).save(model_path)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_name": "word_detector_online",
                "threshold": grouper.threshold,
                "feature_names": PAIR_FEATURE_NAMES,
                "training": training,
            },
            f,
            indent=2,
        )
    return {"model": model_path, "config": config_path}


def export(output_dir: Path) -> dict[str, Path]:
    """Train on IAM-OnDB and write the artifact files to ``output_dir``."""
    pages, _ = load_iam_ondb_pages(TRAINING_FORMS, gap_squeeze=TRAINING_GAP_SQUEEZE)
    grouper = GROUPERS[DETECTOR]().fit(pages)
    paths = write_artifacts(
        grouper,
        output_dir,
        training={
            "dataset": "IAM-OnDB",
            "forms": TRAINING_FORMS,
            "gap_squeeze": list(TRAINING_GAP_SQUEEZE),
            "detector": DETECTOR,
            "code_revision": _git_sha(),
        },
    )
    print(f"Wrote {paths['model']} and {paths['config']}")
    return paths


def upload_to_hub(output_dir: Path, repo_id: str = HF_REPO_ID) -> None:
    """Upload the artifact files and the model card to HF Hub (ADR 006 section 1)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    for local, name in (
        (output_dir / "model.npz", "model.npz"),
        (output_dir / "config.json", "config.json"),
        (MODEL_CARD, "README.md"),
    ):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=repo_id)
    print(f"Uploaded model.npz, config.json and README.md to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory to write model.npz and config.json into.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After export, upload to HF Hub (requires authentication).",
    )
    args = parser.parse_args()

    export(args.output_dir)
    if args.upload:
        upload_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
