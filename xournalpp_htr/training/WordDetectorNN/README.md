---
title: Xournal++ HTR WordDetectorNN
emoji: 📄
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: demo.py
pinned: false
---

# Training WordDetectorNN

[🤗 demo](https://huggingface.co/spaces/PellelNitram/xournalpp_htr_WordDetectorNN)

This subfolder contains the standalone training code and resources for training the WordDetectorNN model.
The [WordDetectorNN](https://github.com/githubharald/WordDetectorNN) model was originally created by
[Harald Scheidl](https://github.com/githubharald/WordDetectorNN) & this work here just reimplements it
with some best practises to later integrate it into a Xournal++ HTR pipeline.

## Project Structure

This subfolder operates as an independent module within the main Xournal++ HTR repository. It has been designed to be completely self-contained to enable:

- Simple experimentation without complex dependency management
- Rapid prototyping and iteration
- Isolated development without affecting the main repository

### Why Independent?

The decision to keep this as a standalone subfolder (rather than integrating it directly into the main repository) allows for:

- Streamlined experimentation
- Avoiding dependency conflicts with the main project
- Faster development cycles
- Cleaner separation of concerns during the research phase

## Installation

1. `uv init --no-workspace`
2. `uv venv`
3. `uv sync`; by the way, here is a [useful tutorial](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) on how to install pytorch w/ uv.

## Current Status

Everything from the original WordDetectorNN model has been reimplemented except for training data augmentations.
It is a reasonable idea to implement these training data augmentations in the future. After this is done, the
betham sample should be checked again for correctness as this example fails horrible currently.

This folder remains independent from the main code base until WordDetectorNN becomes part of a pipeline and, before
that, Xournal++ HTR can be installed both for users and developers properly.

## Future Integration

After successful model training, integration work will include:

- Dependency alignment with the main project structure
- Integrate training & inference code into main code base so that it is usable as pipeline

## Deployment as Hugging Face Space

The model implemented and trained here is currently deployed as HF Gradio Space [here](https://huggingface.co/spaces/PellelNitram/xournalpp_htr_WordDetectorNN).

The deployment to there is currently done as manual process. The files are copied to the Space manually.

In the future, it is worth to use a Docker HF space to gain
finer grained control about the deployment process and to automate it.

## Outlook

Considerations for when the model is integrated into a pipeline:

- Train using data augmentations.
- Use PIL images everywhere instead of numpy to keep track of channel order.