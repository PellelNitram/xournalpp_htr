<div align="center">

# Xournal++ HTR

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Developing [handwritten text recognition](https://en.wikipedia.org/wiki/Handwriting_recognition) for [Xournal++](https://github.com/xournalpp/xournalpp).

*Your contributions are greatly appreciated!*

</div>

## Xournal++ HTR in 90 seconds

<div align="center">

[![YouTube](http://i.ytimg.com/vi/boXm7lPFSRQ/hqdefault.jpg)](https://www.youtube.com/watch?v=boXm7lPFSRQ)

*([Click on image or here to get to video](https://www.youtube.com/watch?v=boXm7lPFSRQ?utm_source=github&utm_medium=readme&utm_campaign=github_readme).)*

</div>

## Installation

This project consists of both the inference and training code. Most users will only be interested in the inference part, so that the below only comprises of the inference part that you need to execute the plugin from within Xournal++.

The training part is optional and allows to help to train our own models which improve over time. These installation process is optional and detailed further below.

### Cross-platform

Execute the following commands:

1. Create an environment: ``conda create --name xournalpp_htr python=3.10.11``.
2. Use this environment: ``conda activate xournalpp_htr``.
3. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package using [its installation guide](https://github.com/githubharald/HTRPipeline/tree/master#installation).
4. Install all dependencies of this package ``pip install -r requirements.txt``.
4. Install the package in development mode with ``pip install -e .`` (do not forget the dot, '.').
4. Install pre-commit hooks with: `pre-commit install`.
5. Move `plugin/` folder content to `${XOURNAL_CONFIG_PATH}/plugins/xournalpp_htr/` with `${XOURNAL_CONFIG_PATH}` being the configuration path of Xournal++, see Xournal++ manual [here](https://xournalpp.github.io/guide/file-locations/).
6. Edit `config.lua`, setting `_M.python_executable` to your python executable **in the conda environment** and `_M.xournalpp_htr_path` to the absolute path of this repo. See [the example config](plugin/config.lua) for details.
7. Ensure Xournal++ is on your `PATH`. See [here](https://xournalpp.github.io/guide/file-locations/) for the binary location.

### Linux

Run `bash INSTALL_LINUX.sh` from repository root directory.

This script also installs the plugin as explained in the last point of the cross-platform installation procedure. The installation of the plugin is performed with `plugin/copy_to_plugin_folder.sh`, which can also be invoked independently of `INSTALL_LINUX.sh` for updating the plugin installation.

### After installation

Confirm that the installation worked by running `make tests-installation` from repository root directory.

## Usage

Details relevant for usage of the plugin:

1. Make sure to save your file beforehand. The plugin will also let you know that you
   need to save your file first.
2. After installation, navigate to `Plugin > Xournal++ HTR` to invoke the plugin. Then
   select a filename and press `Save`. Lastly, wait a wee bit until the process is
   finished; the Xournal++ UI will block while the plugin applies HTR to your file.
   If you opened Xournal++ through a command-line, you can see progress bars that show
   the HTR process in real-time.

Note: Currently, the Xournal++ HTR plugin requires you to use a nightly build of
Xournal++ because it uses upstream Lua API features that are not yet part of the
stable build. Using the officially provided Nightly AppImag, see
[here](https://xournalpp.github.io/installation/linux/), is very convenient.
The plugin has been tested with the following nightly Linux build of Xournal++:

```
xournalpp 1.2.3+dev (583a4e47)
└──libgtk: 3.24.20
```

Details relevant for development of the plugin:

1. Activate environment: ``conda activate xournalpp_htr``. Alternatively use ``source activate_env.sh`` as shortcut.
2. Use the code.
3. To update the requirements file: ``pip freeze > requirements.txt``.

## Project description

Taking handwritten notes digitally comes with many benefits but lacks searchability of your notes. Hence, there is a need to make your handwritten notes searchable. This can be achieved with ["handwritten text recognition" (HTR)](https://en.wikipedia.org/wiki/Handwriting_recognition), which is the process of assigning searchable text to written strokes.

While many commercial note taking apps feature great HTR systems to make your notes searchable and there are a number of existing open-source implementations of various algorithms, there is no HTR feature available in an open-source note taking application that is privacy aware and processes your data locally.

<div align="center">
    <p><b>The purpose of the <i>Xournal++ HTR</i> project is to change that!</b></p>
</div>

Xournal++ HTR strives to bring open-source on-device handwriting recognition to [Xournal++](https://github.com/xournalpp/xournalpp) as it is one of the most adopted open-source note taking apps and thereby HTR can be delivered to the largest possible amount of users.

## Training

### Installation

Follow the above installation procedure and replace the step `pip install -r requirements.txt` by both `pip install -r requirements.txt` and `pip install -r requirements_training.txt` to install both the inference and training dependencies.

## Project design

The design of Xournal++ HTR tries to bridge the gap between both delivering a production ready product and allowing contributors to experiment with new algorithms.

The project design involves a Lua plugin and a Python backend, see the following figure. First, the production ready product is delivered by means of an Xournal++ plugin. The plugin is fully integrated in Xournal++ and calls a Python backend that performs the actual transcription. The Python backend allows selection of various recognition models and is thereby fully extendable with new models.

<div align="center">
    <img src="docs/images/system_design.jpg" width="50%">
    <p><i>Design of xournalpp_htr.</i></p>
</div>

Developing a usable HTR systems requires experimentation. The project structure is set up to accommodate this need. *Note that ideas on improved project structures are appreciated.*

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

Feel free to play around with the demo yourself using [this code](https://github.com/PellelNitram/xournalpp_htr/blob/master/scripts/demo_concept_1.sh) after [installing this project](#Installation). The "concept 1" is also what is currently used in the plugin and shown in the [90 seconds demo](https://www.youtube.com/watch?v=boXm7lPFSRQ).

Next steps to improve the performance of the handwritten text recognition even further could be:
- Re-train the algorithm on Xournal++ specific data, while potentially using data augmentation.
- Use language model to improve text encoding.
- Use sequence-to-sequence algorithm that makes use of [Xournal++](https://github.com/xournalpp/xournalpp)'s data format. This translates into using online HTR algorithms.

I would like to acknowledge [Harald Scheidl](https://github.com/githubharald) in this concept as he wrote the underlying algorithms and made them easily usable through [his HTRPipeline repository](https://github.com/githubharald/HTRPipeline) - after all I just feed his algorithm [Xournal++](https://github.com/xournalpp/xournalpp) data in concept 1. [Go check out his great content](https://githubharald.github.io/)!

## Code quality

We try to keep up code quality as high as practically possible. For that reason, the following steps are implemented:

- Testing. Xournal++ HTR uses `pytest` for implementing unit, regression and integration tests.
- Linting. Xournal++ HTR uses `ruff` for linting and code best practises. `ruff` is implemented as git pre-commit hook. Since `ruff` as pre-commit hook is configured externally with `pyproject.toml`, you can use the same settings in your IDE if you wish to speed up the process.
- Formatting. Xournal++ HTR uses `ruff-format` for consistent code formatting. `ruff-format` is implemented as git pre-commit hook. Since `ruff-format` as pre-commit hook is configured externally with `pyproject.toml`, you can use the same settings in your IDE if you wish to speed up the process.

## Community contributions

The following branching strategy is used to keep the `master` branch stable and
allow for experimentation: `master` > `dev` > `feature branches`. This branching
strategy is shown in the following visualisation and then explained in more detail
in the next paragraph:

```mermaid
%%{init:{  "gitGraph":{ "mainBranchName":"master" }}}%%
gitGraph
    commit
    commit
    branch dev
    commit
    checkout dev
    commit
    commit
    branch feature/awesome_new_feature
    commit
    checkout feature/awesome_new_feature
    commit
    commit
    commit
    checkout dev
    merge feature/awesome_new_feature
    commit
    commit
    checkout master
    merge dev
    commit
    commit
```

In more details, this repository adheres to the following git branching strategy: The
`master` branch remains stable and delivers a functioning product. The `dev` branch
consists of all code that will be merged to `master` eventually where the corresponding
features are developed in individual feature branches; the above visualisation shows an
example feature branch called `feature/awesome_new_feature` that works on a feature
called `awesome_new_feature`.

Given this structure, please implement new features as feature branches and
rebase them onto the `dev` branch prior to sending a pull request to `dev`.

Note: The Github Actions CI/CD pipeline runs on the branches `master` and `dev`.

## Acknowledgements

I would like to thank [Leonard Salewski](https://twitter.com/L_Salewski) and [Jonathan Prexl](https://scholar.google.com/citations?user=pqep1wkAAAAJ&hl=en) for useful discussions, [Harald Scheidl](https://github.com/githubharald/) for making his repositories about handwritten text recognition public ([SimpleHTR](https://github.com/githubharald/SimpleHTR), [WordDetectorNN](https://github.com/githubharald/WordDetectorNN) and [HTRPipeline](https://github.com/githubharald/HTRPipeline)) and the [School of Physics and Astronomy](https://www.ph.ed.ac.uk/) at [The University of Edinburgh](https://www.ed.ac.uk/) for providing compute power.

## Cite

If you are using Xournal++ HTR for your research, I'd appreciate if you could cite it. Use:

```
@software{Lellep_Xournalpp_HTR,
  author = {Lellep, Martin},
  title = {xournalpp_htr},
  url = {https://github.com/PellelNitram/xournalpp_htr},
  license = {GPL-2.0},
}
```
