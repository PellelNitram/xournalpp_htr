# ADR 002 – Use Hugging Face Ecosystem for Machine Learning

- Date: 2025-11-02
- Status: ✅ Accepted

## Context

The project originally used plain PyTorch with local files for datasets and model storage.
This made it hard to share and version models; once trained, they essentially lived on a hard drive with no central management or deployment integration and I had to rely on Google Drive and Dropbox links.

## Decision

Adopt the **[Hugging Face ecosystem](https://huggingface.co/)** for all machine learning–related components, including:

* Model Hub: hosting and versioning trained models
* Dataset Hub: storing and sharing datasets
* `transformers` and `datasets` libraries: for training and data handling
* Trainer API: for standard training workflows

## Rationale

Hugging Face offers a free, community-maintained platform that is now the **industry standard** for open ML projects.
It provides built-in **versioning**, **sharing**, and **deployment integration**, making it easy to pull models directly in demos or end-user environments.

## Consequences

### Pros

* Centralized and versioned model/dataset hosting
* Easier sharing, collaboration, and reproducibility
* Straightforward integration in deployments
* Large and active community support
* One can either fully integrate the model by subclassing `PreTrainedModel` or use it as plain artifact storage of
  the binary weights file

### Cons

* Requires learning new APIs and conventions
* Custom training routines may need workarounds
* I'll need to look into how to convert a PyTorch model into a HF model, incl pre and post processing code

## Alternatives

Continuing with plain PyTorch and local storage would have been simpler but lacked any versioning, reproducibility, or sharing capabilities.