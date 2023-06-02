# Xournal++ HTR

Developing handwritten text recognition for [Xournal++](https://github.com/xournalpp/xournalpp).

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