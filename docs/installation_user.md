# Installation

This project consists of both the inference and training code. Most users will only be interested in the inference part, so that the below only comprises of the inference part that you need to execute the plugin from within Xournal++.

The training part is optional and allows to help to train our own models which improve over time. This installation process is optional and detailed in [the developer guide](developer_guide.md#Installation).

## Linux

Run `bash INSTALL_LINUX.sh` from repository root directory.

This script also installs the plugin as explained in the last point of the cross-platform installation procedure. The installation of the plugin is performed with `plugin/copy_to_plugin_folder.sh`, which can also be invoked independently of `INSTALL_LINUX.sh` for updating the plugin installation.

## Cross-platform

If you want to install the plugin manually, then execute the following commands:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package into `external/htr_pipeline/HTRPipeline` (see `INSTALL_LINUX.sh` for the exact steps).
3. Install all dependencies with `uv sync`.
4. Run the configuration script with `uv run xournalpp-htr-configure`.
5. Install pre-commit hooks with: `uv run pre-commit install`.
6. Copy `plugin/` folder content to `${XOURNAL_CONFIG_PATH}/plugins/xournalpp_htr/` with `${XOURNAL_CONFIG_PATH}` being the configuration path of Xournal++, see Xournal++ manual [here](https://xournalpp.github.io/guide/file-locations/).
7. Edit `config.lua`, setting `_M.python_executable` to the uv-managed Python executable and `_M.xournalpp_htr_path` to the absolute path of this repo. See the example config for details in `plugin/config.lua`.
8. Ensure Xournal++ is on your `PATH`. See [here](https://xournalpp.github.io/guide/file-locations/) for the binary location.

## After installation

Confirm that the installation worked by running `make tests-installation` from repository root directory.