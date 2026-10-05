"""Build a word list for lexicon-biased beam search decoding (issue #120).

Unions the IAM training vocabulary (domain-matched, but limited to what IAM
happens to contain) with a general English wordlist (broader coverage, but
not domain-specific), so the beam decoder can bias towards real words without
a trained KenLM language model. Requires the ``training-simple-htr`` extra.

The published list is downloaded by ``SimpleHTRModel.load_vocabulary()`` for
the ``2026-10-05_yolo_detector_beam_vocab`` pipeline; ``--upload`` publishes
it as ``vocabulary.txt`` next to the SimpleHTR model on HF Hub.

    uv run python -m xournalpp_htr.training.simple_htr.build_vocabulary [--upload]
"""

import argparse
import re
from pathlib import Path

from english_words import get_english_words_set

from xournalpp_htr.inference_models import SimpleHTRModel
from xournalpp_htr.training.simple_htr.dataset import IAM_Words_Dataset
from xournalpp_htr.xio import load_IAM_DB_dataset

_WORD_RE = re.compile(r"^[A-Za-z']+$")


def build_vocabulary(data_path: Path | None) -> list[str]:
    """Union of the IAM transcription vocabulary and a general English wordlist."""
    if data_path is None:
        data_path = load_IAM_DB_dataset()

    words_file = data_path / "ascii" / "words.txt"
    iam_entries = IAM_Words_Dataset._parse_words_file(words_file)
    iam_words = {text for _, text in iam_entries}

    general_words = get_english_words_set(["web2"], lower=True)
    general_words |= {w.capitalize() for w in general_words}

    vocabulary = {w for w in iam_words | general_words if _WORD_RE.match(w)}
    return sorted(vocabulary)


def upload_to_hub(vocabulary_file: Path, repo_id: str = SimpleHTRModel.HF_REPO_ID):
    """Publish the word list as ``vocabulary.txt`` next to the SimpleHTR model."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(vocabulary_file),
        path_in_repo="vocabulary.txt",
        repo_id=repo_id,
    )
    print(f"Uploaded {vocabulary_file} as vocabulary.txt to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the IAM dataset root. Defaults to resolving it from HF Hub.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vocabulary.txt"),
        help="Where to write the word list, one word per line.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After building, upload to HF Hub (requires authentication).",
    )
    args = parser.parse_args()

    vocabulary = build_vocabulary(args.data_path)
    args.output.write_text("\n".join(vocabulary) + "\n")
    print(f"Wrote {len(vocabulary)} words to {args.output}")

    if args.upload:
        upload_to_hub(args.output)


if __name__ == "__main__":
    main()
