# TODO!

# PyInstaller Experiment

For easier installation.

Scope: On Linux.

Commands I experimented with after the package has been installed:

```bash
cd xournalpp_htr
pyinstaller --onefile --add-data "../external/htr_pipeline/HTRPipeline/htr_pipeline/models:htr_pipeline/models" --hidden-import "PIL._tkinter_finder" run_htr.py
dist/run_htr --input-file /home/martin/data/xournalpp_htr/test_1.xoj --output-file /home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp-3.pdf
```

This seems to work on my Ubuntu PC.

Open questions:
- Does it work on other linux computers?
    - Idea: check w/ EC2/GCP-VM instances.
    - It does need xournalpp installed to render the document to PDF prior to HTR'ing it. But that's
      a reasonable dependency b/c we're dealing with X++ files here. Note, however, that xournalpp
      must be available as `xournalpp` for the script to work.
      Maybe do `xournalp=<AppImage path> dist/run_htr [.. from above]`?
- How to include the `xournalpp` binary in order to export the `xopp` file to a PDF?
    - Idea: Let the use select the `xournalpp` path?


Add video demo here? -> yes!

Next steps:
- use github actions to produce binaries for MacOS and Windows.



from Github issue:

# Description

To improve the user experience and make `xournalpp_htr` more accessible, this issue proposes a proof of concept for bundling the application into a single, standalone executable using [PyInstaller](https://pyinstaller.org/en/stable/).

# Motivation

Currently, users need to have Python and `conda` set up to install the package and its dependencies on Linux. This can be a barrier for many potential users. A single-file executable would simplify this process to just downloading and running a file.

# Goals for this POC

- Explore if PyInstaller is a viable option.
- If so:
  - Create a build script for PyInstaller.
  - Test that the bundled executable runs correctly without an external Python installation.
- Add findings to documentation

This would make distribution and installation simple and user-friendly.

# Potential future steps

 Build a single executable not only for Linux but also include Windows and MacOS. This can probably be achieved using Github Action.