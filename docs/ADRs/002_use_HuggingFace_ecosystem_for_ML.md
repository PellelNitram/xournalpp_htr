# ADR 002 – Use Hugging Face Ecosystem for Machine Learning

- Date: 2025-11-03
- Status: Accepted
- PRD: None
- Drivers: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))
- Deciders: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))

## Context

The project originally used plain PyTorch with local files for datasets and model storage.
This made it hard to share and version models; once trained, they essentially lived on a hard drive with no central management or deployment integration and I had to rely on Google Drive and Dropbox links.
Also, training models on different machines is cumbersome because one needs to manually download the dataset every time.

## Decision

Adopt the **[Hugging Face ecosystem](https://huggingface.co/)** for all machine learning–related components, including:

* Model Hub: hosting and versioning trained models
* Dataset Hub: storing and sharing datasets
* `transformers` and `datasets` libraries: for training and data handling
* Trainer API: for standard training workflows

Note: We need to agree on a good naming scheme for storing and retrieving models and datasets efficiently on HuggingFace.
This will be the subject of a future ADR.

## Rationale

Hugging Face offers a free, community-maintained platform that is now the industry standard for open ML projects.
It provides built-in versioning and sharing, making it easy to pull models directly in demos or end-user environments.
The same applies to retrieving properly versioned datasets for training, benchmarking, and various demo use cases (e.g., providing sample data in Gradio applications).

## Consequences

### Pros

* Centralized and versioned model/dataset hosting
* Easier sharing, collaboration, and reproducibility
* Straightforward integration in deployments by letting HuggingFace download the model automatically
* Large and active community support
* One can either fully integrate the model by subclassing `PreTrainedModel` or use it as plain artifact storage of
  the binary weights file

### Cons

* Requires learning new APIs and conventions
* Custom training routines may need workarounds
* Invest time to learn how to convert a PyTorch model into a HF model, incl pre and post processing code

## Alternatives

Continuing with plain PyTorch and local storage would have been simpler but lacked any versioning, reproducibility, or sharing capabilities.