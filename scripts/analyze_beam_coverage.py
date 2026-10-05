"""Diagnose whether beam search can even find the words greedy decoding misses.

For every ground-truth/prediction pair where recognition is wrong (issue
#120), this re-decodes the same crop with a wide beam (no language model) and
checks whether the correct word shows up anywhere in the beam. If it rarely
does, the SimpleHTR acoustic model itself is the ceiling -- no amount of beam
search or language-model rescoring can promote a candidate that was never
generated. If it usually does, there is headroom for an LM or lexicon bias to
help by re-ranking existing candidates.

    uv run python scripts/analyze_beam_coverage.py --beam-width 500
"""

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from xournalpp_htr.benchmark import _cer, _load_gt_words, _match
from xournalpp_htr.documents import get_document
from xournalpp_htr.inference_models import SimpleHTRModel, YOLOWordDetectorModel
from xournalpp_htr.models import WordPrediction
from xournalpp_htr.training.shared.ctc_decoding import build_beam_decoder
from xournalpp_htr.xio import load_benchmark

RENDER_DPI = 150


def _detect_words_with_crops(
    document, detector: YOLOWordDetectorModel
) -> tuple[dict[int, list[WordPrediction]], dict[int, np.ndarray]]:
    """Detect word boxes, mirroring the 2026-09-02_yolo_detector branch of
    `compute_predictions`, but also return each crop (`recognize()` alone
    doesn't expose it) so it can be re-decoded with a wide beam.
    """
    predictions: dict[int, list[WordPrediction]] = {}
    crops: dict[int, np.ndarray] = {}  # id(WordPrediction) -> crop

    for page_index in range(len(document.pages)):
        with tempfile.NamedTemporaryFile(
            dir="/tmp",
            delete=False,
            prefix=f"xournalpp_htr__page{page_index}__",
            suffix=".jpg",
        ) as tmpfile:
            tmp_path = Path(tmpfile.name)
        written_file = document.save_page_as_image(
            page_index, tmp_path, False, dpi=RENDER_DPI
        )

        if (
            len(document.pages[page_index].layers) == 0
            or len(document.pages[page_index].layers[0].strokes) == 0
        ):
            predictions[page_index] = []
            continue

        img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
        boxes = detector.detect(img)
        coord_scale = document.DPI / RENDER_DPI

        page_predictions = []
        for box in boxes:
            x_min = max(0, int(box.x_min))
            y_min = max(0, int(box.y_min))
            x_max = min(img.shape[1], int(box.x_max))
            y_max = min(img.shape[0], int(box.y_max))
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
            pred = WordPrediction(
                text="",  # filled in by the caller after recognition
                xmin=box.x_min * coord_scale,
                xmax=box.x_max * coord_scale,
                ymin=box.y_min * coord_scale,
                ymax=box.y_max * coord_scale,
            )
            crops[id(pred)] = crop
            page_predictions.append(pred)
        predictions[page_index] = page_predictions

    return predictions, crops


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=500,
        help="Beam width for the reachability check (no language model).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Git tag or commit hash for the benchmark dataset revision.",
    )
    args = parser.parse_args()

    detector = YOLOWordDetectorModel.from_pretrained()
    recognizer = SimpleHTRModel.from_pretrained()
    beam_decoder = build_beam_decoder(recognizer._charset)

    reachable = 0
    unreachable = 0
    examples_unreachable: list[tuple[str, str]] = []
    examples_reachable: list[tuple[str, str]] = []

    for sample in load_benchmark(dataset_version=args.dataset_version):
        document = get_document(sample.xopp_path)
        gt_words = _load_gt_words(sample.gt_path, document)
        predictions, crops = _detect_words_with_crops(document, detector)

        for page_preds in predictions.values():
            for pred in page_preds:
                pred.text = recognizer.recognize(crops[id(pred)])

        pairs = _match(gt_words, predictions)

        for gt_word, pred_word in pairs:
            cer_ci = _cer(gt_word.text.lower(), pred_word.text.lower())
            if cer_ci == 0.0:
                continue  # already correct, nothing to diagnose

            crop = crops[id(pred_word)]
            log_probs = recognizer._compute_log_probs(crop)
            beams = beam_decoder.decode_beams(log_probs, beam_width=args.beam_width)
            beam_texts = {beam[0].lower() for beam in beams}

            if gt_word.text.lower() in beam_texts:
                reachable += 1
                if len(examples_reachable) < 5:
                    examples_reachable.append((gt_word.text, pred_word.text))
            else:
                unreachable += 1
                if len(examples_unreachable) < 5:
                    examples_unreachable.append((gt_word.text, pred_word.text))

    total = reachable + unreachable
    print(f"Mismatched words analysed: {total}")
    if total == 0:
        return
    print(
        f"  Reachable in beam width {args.beam_width}:   {reachable} ({reachable / total:.1%})"
    )
    print(
        f"  Never appears in the beam:      {unreachable} ({unreachable / total:.1%})"
    )

    if examples_reachable:
        print(
            "\nExamples where the correct word IS in the beam (greedy just picked wrong):"
        )
        for gt, pred in examples_reachable:
            print(f"  GT={gt!r:20} greedy={pred!r}")

    if examples_unreachable:
        print("\nExamples where the correct word is NEVER generated by the network:")
        for gt, pred in examples_unreachable:
            print(f"  GT={gt!r:20} greedy={pred!r}")


if __name__ == "__main__":
    main()
