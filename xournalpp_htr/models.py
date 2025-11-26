# Add code related to models here. Mostly for inference as there will exist
# another module for training or loading custom models.

import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import DetectorConfig, LineClusteringConfig, read_page
from tqdm import tqdm


def compute_predictions(model_name: str, document) -> dict:
    predictions = {}

    if model_name == "2024-07-18_htr_pipeline":
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
                    page_index, TMP_FILE, False, dpi=150
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
                # height = 700 # good
                # enlarge = 5
                # enlarge = 10
                # height = 1000 # good
                # height = 1600 # not good
                scale = 0.4
                margin = 5
                read_lines = read_page(
                    img,
                    DetectorConfig(scale=scale, margin=margin),
                    line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                )

                predictions_page = []
                for line in read_lines:
                    for word in line:
                        data = {
                            "page_index": page_index,
                            "text": word.text,
                            "xmin": word.aabb.xmin,
                            "xmax": word.aabb.xmax,
                            "ymin": word.aabb.ymin,
                            "ymax": word.aabb.ymax,
                        }
                        predictions_page.append(data)
                predictions[page_index] = predictions_page

    else:
        raise NotImplementedError(f'Model "{model_name}" not implemented.')

    return predictions


def store_predictions_as_images(
    output_directory: Path, predictions: dict, document
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    nr_pages = len(document.pages)

    for page_index in tqdm(range(nr_pages), desc="Store predictions as images"):
        file_name = output_directory / f"page{page_index}.jpg"
        file_name_ocrd = output_directory / f"page{page_index}_ocrd.jpg"

        written_file = document.save_page_as_image(
            page_index, file_name, False, dpi=150
        )

        # ======
        # Do HTR
        # ======

        # read image
        img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

        # To prepare plotting
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Impose predictions on image
        for prediction in predictions[page_index]:
            text = prediction["text"]
            xmin = prediction["xmin"]
            xmax = prediction["xmax"]
            ymin = prediction["ymin"]
            ymax = prediction["ymax"]

            img = cv2.rectangle(
                img, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 2
            )

            img = cv2.putText(
                img,
                text=text,
                org=(int(xmin), int(ymin)),
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
