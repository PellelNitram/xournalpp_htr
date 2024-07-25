from pathlib import Path

import pytest

from xournalpp_htr.run_htr import parse_arguments
from xournalpp_htr.run_htr import main


def test_parse_arguments_empty():
    with pytest.raises(SystemExit) as e_info:
        parse_arguments()

def test_parse_arguments_full():
    args = parse_arguments('-if input -of output -m dummy -pid dir -sp')
    assert len(args) == 5
    assert args['input_file'].stem == 'input'
    assert args['output_file'].stem == 'output'
    assert args['model'] == 'dummy'
    assert args['prediction_image_dir'].stem == 'dir'
    assert args['show_predictions'] == True