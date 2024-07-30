import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import DetectorConfig, LineClusteringConfig, read_page

from xournalpp_htr.documents import XournalDocument


def parse_arguments():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-if",
        "--input-file",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to the input file.",
    )
    args = vars(parser.parse_args())
    return args


def main(args):
    TMP_FILE = Path.home() / Path("xournalpp_htr_tmp_image.jpg")

    file_ending = args["input_file"].suffix

    if file_ending == ".xoj":
        document = XournalDocument(args["input_file"])
    else:
        raise NotImplementedError(
            f'File ending "{file_ending}" currently not readable.'
        )

    nr_pages = len(document.pages)

    for page_index in range(nr_pages):
        page_str = f"Page {page_index+1} / {nr_pages}"
        print()
        print("=" * len(page_str))
        print(page_str)
        print("=" * len(page_str))

        print()
        print("I recognised:")
        print()

        # ==============
        # Write document
        # ==============

        written_file = document.save_page_as_image(page_index, TMP_FILE, False, dpi=150)

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

        # To prepare plotting
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # output text
        for read_line in read_lines:
            recognitions = " ".join(read_word.text for read_word in read_line)
            print(f'- "{recognitions}"')
            for read_word in read_line:
                box_int = read_word.aabb.enlarge_to_int_grid()

                img = cv2.rectangle(
                    img,
                    (int(box_int.xmin), int(box_int.ymax)),
                    (int(box_int.xmax), int(box_int.ymin)),
                    (255, 0, 0),
                    2,
                )

                img = cv2.putText(
                    img,
                    text=read_word.text,
                    org=(int(box_int.xmin), int(box_int.ymin)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=1,
                )

        plt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(plt_image)
        plt.title(page_str)
        plt.show()

        TMP_FILE.unlink(missing_ok=True)

    # TODO: Export as PDF? Try to write a script that uses XOJ as input and exports a PDF with text layer


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
