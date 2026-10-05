"""Crop analysis: write out what the recognizer saw during a benchmark run.

For every predicted word box this writes:

- ``*_crop.png``  -- the detector crop from the page render,
- ``*_input.png`` -- that crop after SimpleHTR preprocessing (resized and
  centered on the white canvas), i.e. exactly the network input,
- ``*_gt.png``    -- the ground truth word box cut from the same render, for
  comparison (matched words only; GT boxes span stroke centers, so a couple
  of pixels of padding are added to include the ink).

Ground truth words without a matching prediction are written as
``*_missed_*_gt.png``. Each page also gets ``*_overview.png`` with GT boxes in
green and predictions colored by status. ``manifest.csv`` lists every row and
``index.html`` shows them side by side, grouped by status.

Status follows the benchmark's word accuracy, i.e. a matched word is
"correct" when it equals the ground truth case-insensitively.
"""

import csv
import html
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from xournalpp_htr.models import CropRecorder, PageIndex, WordPrediction

if TYPE_CHECKING:
    from xournalpp_htr.benchmark import GroundTruthWord

GT_PAD_PX = 3
STATUS_ORDER = ["wrong", "spurious", "missed", "correct"]
STATUS_COLORS_BGR = {
    "correct": (0, 160, 0),
    "wrong": (0, 0, 255),
    "spurious": (255, 0, 255),
    "missed": (0, 140, 255),
}


def _cut(img: np.ndarray, xmin, ymin, xmax, ymax, pad: int = 0) -> np.ndarray:
    x0 = max(0, int(xmin) - pad)
    y0 = max(0, int(ymin) - pad)
    x1 = min(img.shape[1], int(xmax) + pad)
    y1 = min(img.shape[0], int(ymax) + pad)
    return img[y0:y1, x0:x1]


def _to_px(word, coord_scale: float) -> tuple[float, float, float, float]:
    """Document-unit box of a GT word or prediction -> render pixels."""
    return (
        word.xmin / coord_scale,
        word.ymin / coord_scale,
        word.xmax / coord_scale,
        word.ymax / coord_scale,
    )


def _draw(overview: np.ndarray, word, coord_scale: float, color, thickness: int):
    x0, y0, x1, y1 = _to_px(word, coord_scale)
    cv2.rectangle(overview, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)


class CropAnalysis:
    """Collects crop analysis rows sample by sample, then writes the index."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rows: list[dict] = []
        self.overviews: list[str] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def add_sample(
        self,
        sample_name: str,
        document_dpi: float,
        gt_words: list["GroundTruthWord"],
        predictions: dict[PageIndex, list[WordPrediction]],
        pairs: list[tuple["GroundTruthWord", WordPrediction]],
        recorder: CropRecorder,
    ) -> None:
        """Write the images of one sample, as scored by `run_benchmark`."""
        (self.output_dir / sample_name).mkdir(exist_ok=True)
        gt_for_pred = {id(p): g for g, p in pairs}
        matched_gt = {id(g) for g, _ in pairs}

        for page_index, img in recorder.pages.items():
            coord_scale = document_dpi / recorder.render_dpi
            overview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            page_gt = [g for g in gt_words if g.page_index == page_index]

            for gt in page_gt:
                _draw(overview, gt, coord_scale, (0, 200, 0), 1)

            for i, pred in enumerate(predictions.get(page_index, [])):
                gt = gt_for_pred.get(id(pred))
                if gt is None:
                    status = "spurious"
                elif gt.text.lower() == pred.text.lower():
                    status = "correct"
                else:
                    status = "wrong"

                crop = recorder.crops[id(pred)]
                stem = f"{sample_name}/p{page_index}_{i:03d}_{status}"
                self._write(f"{stem}_crop.png", crop)
                self._write(f"{stem}_input.png", recorder.network_inputs[id(pred)])
                gt_path = ""
                if gt is not None:
                    gt_path = f"{stem}_gt.png"
                    self._write(
                        gt_path, _cut(img, *_to_px(gt, coord_scale), pad=GT_PAD_PX)
                    )

                _draw(overview, pred, coord_scale, STATUS_COLORS_BGR[status], 2)
                cv2.putText(
                    overview,
                    str(i),
                    (int(pred.xmin / coord_scale), int(pred.ymin / coord_scale) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    STATUS_COLORS_BGR[status],
                    1,
                )
                self.rows.append(
                    {
                        "sample": sample_name,
                        "page": page_index,
                        "index": i,
                        "status": status,
                        "gt_text": gt.text if gt is not None else "",
                        "pred_text": pred.text,
                        "crop_path": f"{stem}_crop.png",
                        "input_path": f"{stem}_input.png",
                        "gt_path": gt_path,
                        "crop_height": crop.shape[0],
                        "crop_width": crop.shape[1],
                    }
                )

            missed = [g for g in page_gt if id(g) not in matched_gt]
            for j, gt in enumerate(missed):
                gt_path = f"{sample_name}/p{page_index}_missed_{j:03d}_gt.png"
                self._write(gt_path, _cut(img, *_to_px(gt, coord_scale), pad=GT_PAD_PX))
                _draw(overview, gt, coord_scale, STATUS_COLORS_BGR["missed"], 2)
                self.rows.append(
                    {
                        "sample": sample_name,
                        "page": page_index,
                        "index": f"m{j}",
                        "status": "missed",
                        "gt_text": gt.text,
                        "pred_text": "",
                        "crop_path": "",
                        "input_path": "",
                        "gt_path": gt_path,
                        "crop_height": "",
                        "crop_width": "",
                    }
                )

            overview_path = f"{sample_name}/p{page_index}_overview.png"
            self._write(overview_path, overview)
            self.overviews.append(overview_path)

    def write(self) -> dict[str, int]:
        """Write ``manifest.csv`` and ``index.html``; return counts per status."""
        fieldnames = [
            "sample",
            "page",
            "index",
            "status",
            "gt_text",
            "pred_text",
            "crop_path",
            "input_path",
            "gt_path",
            "crop_height",
            "crop_width",
        ]
        with open(self.output_dir / "manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self._write_html()
        return {s: sum(r["status"] == s for r in self.rows) for s in STATUS_ORDER}

    def _write(self, relative_path: str, img: np.ndarray) -> None:
        cv2.imwrite(str(self.output_dir / relative_path), img)

    def _write_html(self) -> None:
        def img_cell(path: str, css_class: str = "") -> str:
            img = f'<img src="{html.escape(path)}">' if path else ""
            return f'<td class="{css_class}">{img}</td>'

        body = [
            "<!doctype html><meta charset='utf-8'><title>Crop analysis</title>",
            "<style>body{font-family:sans-serif} td{padding:4px 8px;"
            "border-bottom:1px solid #ddd;vertical-align:middle} "
            "img{max-height:90px;image-rendering:pixelated;border:1px solid #ccc} "
            "td.input img{height:64px}</style>",
            "<h1>Crop analysis</h1><h2>Page overviews</h2><ul>",
        ]
        body += [
            f'<li><a href="{html.escape(p)}">{html.escape(p)}</a></li>'
            for p in self.overviews
        ]
        body.append("</ul>")

        for status in STATUS_ORDER:
            status_rows = [r for r in self.rows if r["status"] == status]
            body.append(f"<h2>{status} ({len(status_rows)})</h2><table>")
            body.append(
                "<tr><th>sample / page / #</th><th>GT</th><th>prediction</th>"
                "<th>detector crop</th><th>network input</th><th>GT box</th></tr>"
            )
            for r in status_rows:
                body.append(
                    "<tr>"
                    f"<td>{html.escape(r['sample'])} / p{r['page']} / {r['index']}</td>"
                    f"<td>{html.escape(r['gt_text'])}</td>"
                    f"<td>{html.escape(r['pred_text'])}</td>"
                    f"{img_cell(r['crop_path'])}"
                    f"{img_cell(r['input_path'], 'input')}"
                    f"{img_cell(r['gt_path'])}"
                    "</tr>"
                )
            body.append("</table>")

        (self.output_dir / "index.html").write_text("\n".join(body))
