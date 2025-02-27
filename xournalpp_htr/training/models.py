"""Module concerned with models for training custom xournalpp_htr models."""

from lightning import LightningModule


class WordDetectorNN(LightningModule):
    """
    My implementation of [WordDetectorNN](https://github.com/githubharald/WordDetectorNN/)
    by [Harald Scheidl](Harald Scheidl).

    See [here](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) for information
    on PyTorch Lightning's `LightningModule`.
    """

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
