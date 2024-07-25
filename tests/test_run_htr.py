from pathlib import Path

import pytest

from xournalpp_htr.run_htr import parse_arguments
from xournalpp_htr.run_htr import main


def test_parse_arguments_empty():
    with pytest.raises(SystemExit) as e_info:
        parse_arguments()