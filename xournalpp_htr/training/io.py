"""Generic IO functionality."""

import json
from pathlib import Path


def store_alphabet(outfile: Path, alphabet: list[str]) -> None:
    """Stores the alphabet as JSON.

    :param outfile: The path to store the alphabet under.
    :param alphabet: The alphabet.
    """
    with open(outfile, "w") as f:
        json.dump({"alphabet": alphabet}, f, indent=4)


def load_alphabet(infile: Path) -> list[str]:
    """Load alphabet from JSON.

    :param infile: The path to load the alphabet from.
    :returns: The alphabet as list of strings.
    """
    with open(infile, "r") as f:
        json_data = json.load(f)
    return json_data["alphabet"]
