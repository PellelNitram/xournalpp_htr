from pathlib import Path

import pytest

from xournalpp_htr.run_htr import parse_arguments
from xournalpp_htr.run_htr import main


@pytest.fixture
def get_repo_root_directory(request):
    """TODO.

    Fixture based on [this](https://stackoverflow.com/a/57039134).
    """
    rootdir = Path(request.config.rootdir)
    assert (rootdir / 'README.md').is_file()
    return rootdir

@pytest.mark.installation
def test_parse_arguments_empty():
    with pytest.raises(SystemExit) as e_info:
        parse_arguments()

@pytest.mark.installation
def test_parse_arguments_full():
    args = parse_arguments('-if input -of output -m dummy -pid dir -sp')
    assert len(args) == 5
    assert args['input_file'].stem == 'input'
    assert args['output_file'].stem == 'output'
    assert args['model'] == 'dummy'
    assert args['prediction_image_dir'].stem == 'dir'
    assert args['show_predictions'] == True