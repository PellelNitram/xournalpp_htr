# Add code related to models here. Mostly for inference as there will exist
# another module for training or loading custom models.

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from htr_pipeline import DetectorConfig, LineClusteringConfig, read_page
from tqdm import tqdm

from xournalpp_htr.documents import StrokesDocument
from xournalpp_htr.inference_models import (
    LFM3VLModel,
    OnlineWordDetectorModel,
    PPOCRv5Model,
    RFDETRWordDetectorModel,
    SimpleHTRModel,
    SimpleHTROnDBModel,
    TrOCRLargeModel,
    TrOCRModel,
    WordDetectorModel,
    YOLOWordDetectorModel,
    strip_appended_punctuation,
)

PageIndex = int


@dataclass
class WordPrediction:
    text: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


# Pipeline that detects words by grouping the strokes with the published
# stroke-based detector (issue #168) and reads each word with SimpleHTR-OnDB. Each
# crop is rendered from the strokes of its word alone, so that neighbouring words
# inside its box stay out.
ONLINE_DETECTOR_PIPELINES = {"2026-11-03_online_word_detector_hub_simple_htr_ondb"}


# Padding, in document units, around the strokes of a group when cropping it.
ONLINE_DETECTOR_CROP_PADDING = 3.0


# Pipelines that let LFM3-VL-2B read whole pages and return the word boxes
# (issue #171). The second one then snaps each box to the strokes inside it.
LFM3_VL_PAGE_PIPELINES = {
    "2026-11-09_lfm3_vl_page",
    "2026-11-10_lfm3_vl_page_snapped",
}


# Pipelines that cut word crops and recognise them one by one, and can
# therefore fill a `CropRecorder`.
CROP_RECORDING_PIPELINES = {
    "2026-06-07_htr_pipeline_native",
    "2026-09-02_yolo_detector",
    "2026-09-17_rf_detr_detector",
    "2026-10-05_yolo_detector_beam_vocab",
    "2026-10-05_yolo_detector_trocr",
    "2026-10-05_yolo_detector_trocr_large",
    "2026-10-09_yolo_detector_trocr_large_strip_punct",
    "2026-10-09_yolo_detector_ppocrv5",
    "2026-10-17_yolo_detector_simple_htr_ondb",
    "2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab",
    "2026-11-09_yolo_detector_lfm3_vl",
    *ONLINE_DETECTOR_PIPELINES,
}


@dataclass
class CropRecorder:
    """What a detector + recognizer pipeline fed the recognizer, for crop analysis.

    Crops and network inputs are keyed by ``id()`` of the `WordPrediction`
    returned for them by `compute_predictions`.
    """

    render_dpi: float | None = None
    pages: dict[PageIndex, np.ndarray] = field(default_factory=dict)
    crops: dict[int, np.ndarray] = field(default_factory=dict)
    network_inputs: dict[int, np.ndarray] = field(default_factory=dict)


def snap_predictions_to_strokes(
    predictions: list[WordPrediction],
    strokes: list[tuple[np.ndarray, np.ndarray]],
    min_inside: float = 0.5,
) -> list[WordPrediction]:
    """Replace each predicted box by the bounding box of the strokes inside it.

    Every stroke goes to the box that contains the largest fraction of its
    points (the smaller box on ties), provided that fraction is at least
    `min_inside`. A box that gets no stroke is kept as it is. Boxes from a model
    that only sees pixels are rarely tight on the ink; the strokes are exact.
    """
    if not predictions or not strokes:
        return predictions

    boxes = np.array([[p.xmin, p.ymin, p.xmax, p.ymax] for p in predictions])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Tiny differences in area decide ties, so keep them out of the fractions.
    order = np.argsort(areas, kind="stable")

    assigned: list[list[int]] = [[] for _ in predictions]
    for i, (xs, ys) in enumerate(strokes):
        inside = (
            (xs[:, None] >= boxes[None, :, 0])
            & (xs[:, None] <= boxes[None, :, 2])
            & (ys[:, None] >= boxes[None, :, 1])
            & (ys[:, None] <= boxes[None, :, 3])
        ).mean(axis=0)
        # Walk the boxes smallest first and keep the first maximum.
        best = order[np.argmax(inside[order])]
        if inside[best] >= min_inside:
            assigned[best].append(i)

    snapped = []
    for prediction, indices in zip(predictions, assigned, strict=True):
        if not indices:
            snapped.append(prediction)
            continue
        xs = np.concatenate([strokes[i][0] for i in indices])
        ys = np.concatenate([strokes[i][1] for i in indices])
        snapped.append(
            WordPrediction(
                text=prediction.text,
                xmin=float(xs.min()),
                xmax=float(xs.max()),
                ymin=float(ys.min()),
                ymax=float(ys.max()),
            )
        )
    return snapped


def _render_isolated_strokes(
    strokes: list[tuple[np.ndarray, np.ndarray]],
    box: tuple[float, float, float, float],
    dpi: float,
) -> np.ndarray:
    """Render only `strokes` into a grayscale image of `box` (document units).

    Goes through the same rendering and JPEG round trip as the page renders,
    so the crop looks like one cut out of a page, minus any other strokes.
    """
    xmin, ymin, xmax, ymax = box
    document = StrokesDocument(
        [(x - xmin, y - ymin) for x, y in strokes], xmax - xmin, ymax - ymin
    )
    with tempfile.NamedTemporaryFile(
        dir="/tmp", delete=False, prefix="xournalpp_htr__group__", suffix=".jpg"
    ) as tmpfile:
        path = Path(tmpfile.name)
    document.save_page_as_image(0, path, False, dpi=dpi)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    path.unlink()
    return image


def compute_predictions(
    pipeline_name: str,
    document,
    crop_recorder: CropRecorder | None = None,
) -> dict[PageIndex, list[WordPrediction]]:
    """Run HTR on a document and return word-level predictions.

    Bounding box coordinates are always in document units (72 DPI), regardless
    of the internal rendering resolution used by the pipeline. See ADR 005.

    ``crop_recorder``, if given, is filled with page renders, word crops and
    network inputs by the pipelines in `CROP_RECORDING_PIPELINES`; ignored by
    the others.
    """
    predictions: dict[PageIndex, list[WordPrediction]] = {}

    if pipeline_name == "2024-07-18_htr_pipeline":
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                # Confirm that page is not empty
                # (This is not ideal as it'd be better if the `read_page`
                # function below could handle empty pages gracefully but that
                # function is part of an external package (at least for
                # now) so that I cannot alter it for now.)
                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                # ======
                # Do HTR
                # ======

                # read image
                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

                # detect and read text
                detector_scale = 0.4
                margin = 5
                read_lines = read_page(
                    img,
                    DetectorConfig(scale=detector_scale, margin=margin),
                    line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                )

                # Convert bounding boxes from render pixels to document units.
                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for line in read_lines:
                    for word in line:
                        predictions_page.append(
                            WordPrediction(
                                text=word.text,
                                xmin=word.aabb.xmin * coord_scale,
                                xmax=word.aabb.xmax * coord_scale,
                                ymin=word.aabb.ymin * coord_scale,
                                ymax=word.aabb.ymax * coord_scale,
                            )
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name == "2026-06-07_htr_pipeline_native":
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = WordDetectorModel.from_pretrained()
        recognizer = SimpleHTRModel.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
                if crop_recorder is not None:
                    crop_recorder.render_dpi = RENDER_DPI
                    crop_recorder.pages[page_index] = img

                boxes = detector.detect(img)

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for box in boxes:
                    x_min = max(0, int(box.x_min))
                    y_min = max(0, int(box.y_min))
                    x_max = min(img.shape[1], int(box.x_max))
                    y_max = min(img.shape[0], int(box.y_max))

                    crop = img[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    text = recognizer.recognize(crop)

                    prediction = WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name in (
        "2026-09-02_yolo_detector",
        "2026-10-17_yolo_detector_simple_htr_ondb",
    ):
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = YOLOWordDetectorModel.from_pretrained()
        # The ondb variant swaps in the SimpleHTR retrained on rendered
        # IAM-OnDB crops (issue #150); greedy decoding like the original.
        if pipeline_name == "2026-10-17_yolo_detector_simple_htr_ondb":
            recognizer = SimpleHTROnDBModel.from_pretrained()
        else:
            recognizer = SimpleHTRModel.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
                if crop_recorder is not None:
                    crop_recorder.render_dpi = RENDER_DPI
                    crop_recorder.pages[page_index] = img

                boxes = detector.detect(img)

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for box in boxes:
                    x_min = max(0, int(box.x_min))
                    y_min = max(0, int(box.y_min))
                    x_max = min(img.shape[1], int(box.x_max))
                    y_max = min(img.shape[0], int(box.y_max))

                    crop = img[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    text = recognizer.recognize(crop)

                    prediction = WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name == "2026-09-17_rf_detr_detector":
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = RFDETRWordDetectorModel.from_pretrained()
        recognizer = SimpleHTRModel.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
                if crop_recorder is not None:
                    crop_recorder.render_dpi = RENDER_DPI
                    crop_recorder.pages[page_index] = img

                boxes = detector.detect(img)

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for box in boxes:
                    x_min = max(0, int(box.x_min))
                    y_min = max(0, int(box.y_min))
                    x_max = min(img.shape[1], int(box.x_max))
                    y_max = min(img.shape[0], int(box.y_max))

                    crop = img[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    text = recognizer.recognize(crop)

                    prediction = WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name in (
        "2026-10-05_yolo_detector_beam_vocab",
        "2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab",
    ):
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = YOLOWordDetectorModel.from_pretrained()
        # Same detector as 2026-09-02_yolo_detector, but beam search decoding
        # snapped to the nearest known word (issue #120). The ondb variant
        # swaps in the SimpleHTR retrained on rendered IAM-OnDB crops
        # (issue #150), with the same word list.
        if pipeline_name == "2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab":
            recognizer = SimpleHTROnDBModel.from_pretrained()
        else:
            recognizer = SimpleHTRModel.from_pretrained()
        recognizer.use_lexicon(SimpleHTRModel.load_vocabulary())

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
                if crop_recorder is not None:
                    crop_recorder.render_dpi = RENDER_DPI
                    crop_recorder.pages[page_index] = img

                boxes = detector.detect(img)

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for box in boxes:
                    x_min = max(0, int(box.x_min))
                    y_min = max(0, int(box.y_min))
                    x_max = min(img.shape[1], int(box.x_max))
                    y_max = min(img.shape[0], int(box.y_max))

                    crop = img[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    text = recognizer.recognize(crop, decoder="beam")

                    prediction = WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name in (
        "2026-10-05_yolo_detector_trocr",
        "2026-10-05_yolo_detector_trocr_large",
        "2026-10-09_yolo_detector_trocr_large_strip_punct",
        "2026-10-09_yolo_detector_ppocrv5",
        "2026-11-09_yolo_detector_lfm3_vl",
    ):
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = YOLOWordDetectorModel.from_pretrained()
        # Same detector as 2026-09-02_yolo_detector, but pretrained TrOCR in
        # place of SimpleHTR (issue #156), base or large checkpoint.
        # PP-OCRv5 recognition only (issue #157) takes the same slot, as does
        # LFM3-VL-2B behind a llama-server (issue #171).
        if pipeline_name == "2026-10-05_yolo_detector_trocr":
            recognizer = TrOCRModel.from_pretrained()
        elif pipeline_name == "2026-10-09_yolo_detector_ppocrv5":
            recognizer = PPOCRv5Model.from_pretrained()
        elif pipeline_name == "2026-11-09_yolo_detector_lfm3_vl":
            recognizer = LFM3VLModel.from_pretrained()
        else:
            recognizer = TrOCRLargeModel.from_pretrained()
        strip_punct = (
            pipeline_name == "2026-10-09_yolo_detector_trocr_large_strip_punct"
        )

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
                if crop_recorder is not None:
                    crop_recorder.render_dpi = RENDER_DPI
                    crop_recorder.pages[page_index] = img

                boxes = detector.detect(img)

                kept_boxes = []
                crops = []
                for box in boxes:
                    x_min = max(0, int(box.x_min))
                    y_min = max(0, int(box.y_min))
                    x_max = min(img.shape[1], int(box.x_max))
                    y_max = min(img.shape[0], int(box.y_max))

                    crop = img[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue
                    kept_boxes.append(box)
                    crops.append(crop)

                # TrOCR is a transformer, so recognise a page's crops in
                # batches rather than one at a time.
                texts = recognizer.recognize_batch(crops)
                if strip_punct:
                    texts = [strip_appended_punctuation(text) for text in texts]

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = []
                for box, crop, text in zip(kept_boxes, crops, texts, strict=True):
                    prediction = WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
                predictions[page_index] = predictions_page

    elif pipeline_name == "2026-10-09_ppocrv5_det_rec":
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        # PP-OCRv5's own detector and recogniser on whole pages (issue #157).
        model = PPOCRv5Model.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

                coord_scale = document.DPI / RENDER_DPI
                predictions[page_index] = [
                    WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    for text, box in model.read_page(img)
                ]

    elif pipeline_name in LFM3_VL_PAGE_PIPELINES:
        RENDER_DPI = 150
        nr_pages = len(document.pages)

        # LFM3-VL-2B reads whole pages and returns the word boxes itself, no
        # separate detector (issue #171).
        model = LFM3VLModel.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

                coord_scale = document.DPI / RENDER_DPI
                predictions_page = [
                    WordPrediction(
                        text=text,
                        xmin=box.x_min * coord_scale,
                        xmax=box.x_max * coord_scale,
                        ymin=box.y_min * coord_scale,
                        ymax=box.y_max * coord_scale,
                    )
                    for text, box in model.read_page(img)
                ]
                if pipeline_name == "2026-11-10_lfm3_vl_page_snapped":
                    predictions_page = snap_predictions_to_strokes(
                        predictions_page,
                        [
                            (stroke.x, stroke.y)
                            for layer in document.pages[page_index].layers
                            for stroke in layer.strokes
                        ],
                    )
                predictions[page_index] = predictions_page

    elif pipeline_name == "2026-11-21_rnote_lfm3_vl":
        nr_pages = len(document.pages)

        # Replica of the Rnote fork's handwriting recognition (issue #171): the
        # whole page in one LFM3-VL-2B call, which returns the word boxes. The
        # page is rendered, as there, with its longest edge at about 2048 px.
        model = LFM3VLModel.from_pretrained()

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            page_meta = document.pages[page_index].meta_data
            # Raster pixels per document unit.
            scale = LFM3VLModel.rnote_scale(
                float(page_meta["width"]), float(page_meta["height"])
            )

            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(
                    page_index, TMP_FILE, True, dpi=scale * document.DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

                predictions[page_index] = [
                    WordPrediction(
                        text=text,
                        xmin=box.x_min / scale,
                        xmax=box.x_max / scale,
                        ymin=box.y_min / scale,
                        ymax=box.y_max / scale,
                    )
                    for text, box in model.read_page_like_rnote(img)
                ]

    elif pipeline_name in ONLINE_DETECTOR_PIPELINES:
        # Imported here, as the other pipelines do not need the detector's training code.
        from xournalpp_htr.training.word_detector_online.data import unlabeled_pages

        RENDER_DPI = 150
        nr_pages = len(document.pages)

        detector = OnlineWordDetectorModel.from_pretrained()
        recognizer = SimpleHTROnDBModel.from_pretrained()

        stroke_pages = unlabeled_pages(document)

        for page_index in tqdm(range(nr_pages), desc="Recognition"):
            layers = [p for p in stroke_pages if p.page_index == page_index]
            if not layers:
                print(f"Page {page_index} is empty. Skipping HTR.")
                predictions[page_index] = []
                continue

            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"xournalpp_htr__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                written_file = document.save_page_as_image(
                    page_index, Path(tmpfile.name), True, dpi=RENDER_DPI
                )

            img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
            if crop_recorder is not None:
                crop_recorder.render_dpi = RENDER_DPI
                crop_recorder.pages[page_index] = img

            predictions_page = []
            for layer in layers:
                for group in detector.group_strokes(layer.strokes):
                    xs = np.concatenate([layer.strokes[i][0] for i in group])
                    ys = np.concatenate([layer.strokes[i][1] for i in group])
                    pad = ONLINE_DETECTOR_CROP_PADDING
                    xmin, xmax = xs.min() - pad, xs.max() + pad
                    ymin, ymax = ys.min() - pad, ys.max() + pad

                    crop = _render_isolated_strokes(
                        [layer.strokes[i] for i in sorted(group)],
                        (xmin, ymin, xmax, ymax),
                        RENDER_DPI,
                    )
                    if crop.size == 0:
                        continue

                    prediction = WordPrediction(
                        text=recognizer.recognize(crop),
                        xmin=float(xmin),
                        xmax=float(xmax),
                        ymin=float(ymin),
                        ymax=float(ymax),
                    )
                    predictions_page.append(prediction)
                    if crop_recorder is not None:
                        crop_recorder.crops[id(prediction)] = crop
                        crop_recorder.network_inputs[id(prediction)] = (
                            recognizer.preprocess(crop)
                        )
            predictions[page_index] = predictions_page

    else:
        raise NotImplementedError(f'Pipeline "{pipeline_name}" not implemented.')

    return predictions


def store_predictions_as_images(
    output_directory: Path,
    predictions: dict[PageIndex, list[WordPrediction]],
    document,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    nr_pages = len(document.pages)

    for page_index in tqdm(range(nr_pages), desc="Store predictions as images"):
        file_name = output_directory / f"page{page_index}.jpg"
        file_name_ocrd = output_directory / f"page{page_index}_ocrd.jpg"

        written_file = document.save_page_as_image(
            page_index, file_name, False, dpi=150
        )

        # read image
        img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

        # To prepare plotting
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Impose predictions on image
        for prediction in predictions[page_index]:
            img = cv2.rectangle(
                img,
                (int(prediction.xmin), int(prediction.ymax)),
                (int(prediction.xmax), int(prediction.ymin)),
                (255, 0, 0),
                2,
            )

            img = cv2.putText(
                img,
                text=prediction.text,
                org=(int(prediction.xmin), int(prediction.ymin)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=1,
            )

        plt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        figure_aspect_ratio = float(
            document.pages[page_index].meta_data["height"]
        ) / float(document.pages[page_index].meta_data["width"])
        plt.figure(figsize=(10, 10 * figure_aspect_ratio))
        plt.imshow(plt_image)
        plt.savefig(file_name_ocrd, dpi=150)
        plt.close()
