"""Script to perform HTR on Xournal(++) document."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig

from xournalpp_htr.documents import XournalDocument


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-if', '--input-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the input Xournal or Xournal++ file.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The model to use for handwriting recognition.') # TODO: Introduce dummy model called "test_lua_to_python"; TODO: Register models somehow to allow choice keyword here
    args = vars( parser.parse_args() )
    return args

def main(args):
    
    pass

    # TODO: Agenda
    # 1. export X file to PDF
    # 2. do HTR on X file using model that is specified by `parse_arguments`
    # 3. store prediction as hidden text to PDF file

    # TODO: Define coordinate transforms between X file, prediction and PDF as matrices like in computer graphics.

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
