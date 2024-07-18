# Add code related to models here. Mostly for inference as there will exist
# another module for training or loading custom models.

from pathlib import Path
import tempfile

from tqdm import tqdm
import cv2
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig


def compute_predictions(model_name: str, document) -> dict:

    predictions = {}

    if model_name == '2024-07-18_htr_pipeline':

        nr_pages = len( document.pages )

        for page_index in tqdm(range(nr_pages), desc='Recognition'):

            with tempfile.NamedTemporaryFile(dir='/tmp', delete=False, prefix=f'xournalpp_htr__page{page_index}__', suffix='.jpg') as tmpfile:
                TMP_FILE = Path(tmpfile.name)

                written_file = document.save_page_as_image(page_index, TMP_FILE, False, dpi=150)

                # ======
                # Do HTR
                # ======

                # read image
                img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

                # detect and read text
                #height = 700 # good
                #enlarge = 5
                #enlarge = 10
                # height = 1000 # good
                # height = 1600 # not good
                scale = 0.4
                margin = 5
                read_lines = read_page(img,
                                       DetectorConfig(scale=scale, margin=margin),
                                       line_clustering_config=LineClusteringConfig(min_words_per_line=2))

                predictions_page = []
                for line in read_lines:
                    for word in line:
                        data = {
                            'page_index': page_index,
                            'text': word.text,
                            'xmin': word.aabb.xmin,
                            'xmax': word.aabb.xmax,
                            'ymin': word.aabb.ymin,
                            'ymax': word.aabb.ymax,
                        }
                        predictions_page.append(data)
                predictions[page_index] = predictions_page

    else:
        raise NotImplementedError(f'Model "{model_name}" not implemented.')

    return predictions