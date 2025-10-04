# TODO!

# PyInstaller Experiment

For easier installation.

Scope: On Linux.

Commands I experimented with:

```bash
cd xournalpp_htr
pyinstaller --onefile --add-data "../external/htr_pipeline/HTRPipeline/htr_pipeline/models:htr_pipeline/models" --hidden-import "PIL._tkinter_finder" run_htr.py
dist/run_htr --input-file /home/martin/data/xournalpp_htr/test_1.xoj --output-file /home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp-3.pdf
```

This seems to work on my Ubuntu PC.

Open questions:
- Does it work on other linux computers?
    - Idea: check w/ EC2/GCP-VM instances.
- How to include the `xournalpp` binary in order to export the `xopp` file to a PDF?
    - Idea: Let the use select the `xournalpp` path?