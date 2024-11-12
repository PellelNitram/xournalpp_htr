# Xournal++ HTR

Developing [handwritten text recognition](https://en.wikipedia.org/wiki/Handwriting_recognition) for [Xournal++](https://github.com/xournalpp/xournalpp).

*Your contributions are greatly appreciated!*

## TODO

Test :-)

TODO: Make this documentation the central part and adapt README for example like [here](https://github.com/jaredpalmer/formik). Replicate sections of current README here therefore.

Note: Remove this section once done.

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