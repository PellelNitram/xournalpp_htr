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
6. Edit `config.lua`, setting `_M.python_executable` to your python executable **in the conda environment** and `_M.xournalpp_htr_path` to the absolute path of this repo. See [the example config](https://github.com/PellelNitram/xournalpp_htr/blob/master/plugin/config.lua) for details.
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

(Also consider starring the project on GitHub.)