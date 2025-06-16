# Training WordDetectorNN

This subfolder contains the training code and resources for training the WordDetectorNN.

## Project Structure

This subfolder operates as an independent module within the main Xournal++ HTR repository. It has been designed to be completely self-contained to enable:

- Simple experimentation without complex dependency management
- Rapid prototyping and iteration
- Isolated development without affecting the main repository

## Installation

1. `uv init --no-workspace`
2. `uv venv`
3. `uv sync`; by the way, here is a [useful tutorial](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) on how to install pytorch w/ uv.

## Current Status

**Development Phase**: This module is currently in active development and remains independent from the main repository structure.

**Integration Plan**: Once the model training is complete, both the trained model and the training code will be integrated back into the main Xournal++ HTR repository.

## Why Independent?

The decision to keep this as a standalone subfolder (rather than integrating it directly into the main repository) allows for:

- Streamlined experimentation
- Avoiding dependency conflicts with the main project
- Faster development cycles
- Cleaner separation of concerns during the research phase

## Future Integration

After successful model training, integration work will include:

- Model deployment into the main repository
- Training pipeline integration
- Dependency alignment with the main project structure

---

*This is a temporary independent structure that will be merged into the main Xournal++ HTR repository upon completion.*