# Xournal++ HTR

Developing [handwritten text recognition](https://en.wikipedia.org/wiki/Handwriting_recognition) for [Xournal++](https://github.com/xournalpp/xournalpp).

## Concept 1

The following shows a demo using real-life handwriting data from a Xournal++ file:

<center>
    <iframe width="560" height="315" src="https://www.youtube.com/embed/FGD_O8brGNY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</center>

Despite not being perfect, the main take away is that the performance is surprisingly good given that the underlying algorithm has not been optimised for Xournal++ data at all. I think the performance is already good enough to be useful for the Xournal++ user base.

Feel free to play around with the demo yourself using [this code](https://github.com/PellelNitram/xournalpp_htr/blob/master/scripts/demo_concept_1.sh) after [installing this project](#Installation).

Next steps to improve the performance of the handwriting text recognition even further could be:
- Re-train the algorithm on Xournal++ specific data, while potentially using data augmentation
- Use language model to improve text encoding
- Use sequence-to-sequence algorithm that makes use of [Xournal++](https://github.com/xournalpp/xournalpp)'s data format. This translates into using online HTR algorithms.

I would like to acknowledge [Harald Scheidl](https://github.com/githubharald) for this concept as he wrote the underlying algorithms for it made them easily usable through [his HTRPipeline repository](https://github.com/githubharald/HTRPipeline). Go check out his great content!

## Installation

1. Create an environment: ``conda create --name xournalpp_htr python=3.10.11``.
2. Use this environmen: ``conda activate xournalpp_htr``.
3. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package using [its installation guide](https://github.com/githubharald/HTRPipeline/tree/master#installation).
4. Install all dependencies of this package ``pip install -r requirements.txt``.
4. Install the package in development mode with ``pip install -e .`` (do not forget the dot, '.').

## Usage

1. Activate environment: ``conda activate xournalpp_htr``. Alternatively use ``source activate_env.sh`` as shortcut.
2. Use the code.
3. To update the requirements file: ``pip freeze > requirements.txt``.