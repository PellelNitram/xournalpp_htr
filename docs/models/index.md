# Models

Documentation for each model trained and shipped as part of Xournal++ HTR.
Each page covers setup, training, evaluation, export and inference for that
model.

- [WordDetector](word_detector.md) — word-level bounding-box detector
  (reimplementation of
  [WordDetectorNN](https://github.com/githubharald/WordDetectorNN)).

New models added in the future should follow the same structure: a page under
`docs/models/<model_name>.md` linked from this index and from the `Models`
section of the navigation. The source lives in
`xournalpp_htr/training/<model_name>/` with a lean `README.md` that refers
back to its docs page.
