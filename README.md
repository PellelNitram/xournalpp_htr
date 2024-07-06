<div align="center">

# Xournal++ HTR

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)

Developing [handwritten text recognition](https://en.wikipedia.org/wiki/Handwriting_recognition) for [Xournal++](https://github.com/xournalpp/xournalpp).

*Your contributions are greatly appreciated!*

</div>

## Xournal++ HTR in 90 seconds

*TODO: Add video here.*

## Project description

Taking handwritten notes digitally comes with many benefits but lacks searchability of your notes. Hence, there is a need to make your handwritten notes searchable. This can be achieved with ["handwritten text recognition" (HTR)](https://en.wikipedia.org/wiki/Handwriting_recognition), which is the process of assigning searchable text to written strokes.

While many commercial note taking apps feature great HTR systems to make your notes searchable and there are a number of existing open-source implementations of various algorithms, there is no HTR feature available in an open-source note taking application that is privacy aware and processes your data locally.

**The purpose of this project is to change that!**

Xournal++ HTR strives to bring open-source on-device handwriting recognition to [Xournal++](https://github.com/xournalpp/xournalpp) as it is one of the most adopted open-source note taking apps and thereby HTR can be delivered to the largest possible amount of users.

<div align="center">
    <img src="docs/images/system_design.jpg" width="50%">
    <p><i>Design of this work.</i></p>
</div>

## Project structure

Developing a usable HTR systems requires experimentation. The project structure is set up to accommodate this need. *Note that ideas on imrpoved project structures are appreciated.*

The experimentation is carried out in terms of "concepts". Each concept explores a different approach to HTR and possibly improves over previous concepts, but not necessarily to allow for freedom in risky experiments. Concept 1 is already implemented and uses a computer vision approach that is explained below.

Future concepts might explore:
- Retrain computer vision models from concept 1 using native data representation of [Xournal++](https://github.com/xournalpp/xournalpp)
- Use sequence-to-sequence models to take advantage of native data representation of [Xournal++](https://github.com/xournalpp/xournalpp)
- Use data augmentation to increase effective size of training data
- Use of language models to correct for spelling mistakes

### Concept 1

This concept uses computer vision based algorithms to first detect words on a page and then to read those words.

The following shows a video demo on YouTube using real-life handwriting data from a Xournal file:

[![Xournal++ HTR - Concept 1 - Demo](https://img.youtube.com/vi/FGD_O8brGNY/0.jpg)](https://www.youtube.com/watch?v=FGD_O8brGNY)

Despite not being perfect, the main take away is that the performance is surprisingly good given that the underlying algorithm has not been optimised for Xournal++ data at all.

**The performance is sufficiently good to be useful for the Xournal++ user base.**

Feel free to play around with the demo yourself using [this code](https://github.com/PellelNitram/xournalpp_htr/blob/master/scripts/demo_concept_1.sh) after [installing this project](#Installation).

Next steps to improve the performance of the handwritten text recognition even further could be:
- Re-train the algorithm on Xournal++ specific data, while potentially using data augmentation.
- Use language model to improve text encoding.
- Use sequence-to-sequence algorithm that makes use of [Xournal++](https://github.com/xournalpp/xournalpp)'s data format. This translates into using online HTR algorithms.

I would like to acknowledge [Harald Scheidl](https://github.com/githubharald) in this concept as he wrote the underlying algorithms and made them easily usable through [his HTRPipeline repository](https://github.com/githubharald/HTRPipeline) - after all I just feed his algorithm [Xournal++](https://github.com/xournalpp/xournalpp) data in concept 1. [Go check out his great content](https://githubharald.github.io/)!

## Installation

On Linux: `bash INSTALL_LINUX.sh`.

The commands executed by the above script are the following ones in case you want to install manually:

1. Create an environment: ``conda create --name xournalpp_htr python=3.10.11``.
2. Use this environment: ``conda activate xournalpp_htr``.
3. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package using [its installation guide](https://github.com/githubharald/HTRPipeline/tree/master#installation).
4. Install all dependencies of this package ``pip install -r requirements.txt``.
4. Install the package in development mode with ``pip install -e .`` (do not forget the dot, '.').

After installation, test the installation by running `make tests-installation` from repository root directory.

## Usage

1. Activate environment: ``conda activate xournalpp_htr``. Alternatively use ``source activate_env.sh`` as shortcut.
2. Use the code.
3. To update the requirements file: ``pip freeze > requirements.txt``.

## Acknowledgements

I would like to thank [Leonard Salewski](https://twitter.com/L_Salewski) and [Jonathan Prexl](https://scholar.google.com/citations?user=pqep1wkAAAAJ&hl=en) for useful discussions, [Harald Scheidl](https://github.com/githubharald/) for making his repositories about handwritten text recognition public ([SimpleHTR](https://github.com/githubharald/SimpleHTR), [WordDetectorNN](https://github.com/githubharald/WordDetectorNN) and [HTRPipeline](https://github.com/githubharald/HTRPipeline)) and the [School of Physics and Astronomy](https://www.ph.ed.ac.uk/) at [The University of Edinburgh](https://www.ed.ac.uk/) for providing compute power.
