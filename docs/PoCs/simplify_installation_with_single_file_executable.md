# PoC 001 – Simplifying Installation with a Single-File Executable

- Date: 2025-11-08  
- Author: Martin Lellep (@PellelNitram)  
- Related Issue: [#34](https://github.com/PellelNitram/xournalpp_htr/issues/34)

## Overview

This proof of concept explores whether [`PyInstaller`](https://pyinstaller.org/en/stable/) can be used to package the `xournalpp_htr` Python application into a single executable file.  
The goal is to simplify installation and usage by removing the need for users to manually install Python or Conda environments.

## Background & Motivation

Currently, `xournalpp_htr` requires a working Python and Conda setup on Linux. This dependency chain can discourage non-technical users.  
A single-file binary distribution would allow users to simply download and run the program — improving accessibility and ease of use.

## Objective

This PoC evaluates whether PyInstaller is a viable option for creating a self-contained executable of `xournalpp_htr` that runs without an external Python installation.

## Experiment Setup

Environment: Ubuntu Linux (local machine)  
Scope: Initial validation of PyInstaller packaging on Linux only.

### Commands Used

From top level folder in repository:

```bash
cd xournalpp_htr
pyinstaller --onefile \
    --add-data "../external/htr_pipeline/HTRPipeline/htr_pipeline/models:htr_pipeline/models" \
    --hidden-import "PIL._tkinter_finder" \
    run_htr.py

dist/run_htr --input-file /home/martin/data/xournalpp_htr/test_1.xoj \
             --output-file /home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp-3.pdf
```

## Results

✅ The generated single-file executable runs successfully on the local Ubuntu machine.
The bundled application performs as expected, including model loading and HTR processing.

## Open Questions

* Cross-system compatibility:

  * Does the executable work on other Linux distributions or minimal environments (e.g., EC2 or GCP VMs)?
  * Next step: Test the binary on a clean VM instance.

* Integration with Xournal++:

  * The tool currently requires `xournalpp` to render `.xopp` documents to PDF prior to recognition.
  * This dependency is natural because we are dealing with xournal++ files here.
  * Possible approaches:

    * Require `xournalpp` to be installed and accessible via `$PATH`.
    * Allow the user to specify a custom `xournalpp` binary (e.g., an AppImage).

* Bundling external dependencies:

  * Could the `xournalpp` binary itself be packaged within the PyInstaller bundle?
  * Requires exploration of size implications and licensing considerations.

## Next Steps

1. Cross-platform builds:

   * Attempt PyInstaller builds for macOS and Windows.
   * Automate using GitHub Actions to produce platform-specific binaries.

2. Integration testing:

   * Validate the binary on clean Linux VMs.
   * Confirm compatibility with `xournalpp` when executed via different paths.

3. Plugin integration:

   * Update the X++ HTR Lua plugin to use the new standalone executable.

4. Documentation update:

   * Document the build and installation process once validated.

## Summary

This PoC confirms that PyInstaller is a viable solution for packaging `xournalpp_htr` into a single-file binary on Linux.
The next phase will focus on cross-platform builds, integration testing, and automating releases to make distribution fully user-friendly.