# Add code related to models here. Mostly for inference as there will exist
# another module for training or loading custom models.

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import DetectorConfig, LineClusteringConfig, read_page
from tqdm import tqdm

from xournalpp_htr.inference_models import SimpleHTRModel, WordDetectorModel

PageIndex = int


@dataclass
class WordPrediction:
    text: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def compute_predictions(
    pipeline_name: str, document
) -> dict[PageIndex, list[WordPrediction]]:
    """Run HTR on a document and return word-level predictions.

    Bounding box coordinates are always in document units (72 DPI), regardless
    of the internal rendering resolution used by the pipeline. See ADR 005.
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
                    page_index, TMP_FILE, False, dpi=RENDER_DPI
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

    elif pipeline_name == "2026-06-07_local_pipeline":
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
                    page_index, TMP_FILE, False, dpi=RENDER_DPI
                )

                if (
                    len(document.pages[page_index].layers) == 0
                    or len(document.pages[page_index].layers[0].strokes) == 0
                ):
                    print(f"Page {page_index} is empty. Skipping HTR.")
                    predictions[page_index] = []
                    continue

                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

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

                    predictions_page.append(
                        WordPrediction(
                            text=text,
                            xmin=box.x_min * coord_scale,
                            xmax=box.x_max * coord_scale,
                            ymin=box.y_min * coord_scale,
                            ymax=box.y_max * coord_scale,
                        )
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
