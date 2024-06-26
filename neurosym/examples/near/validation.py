from typing import List

import torch
import tqdm.auto as tqdm

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.methods.near_example_trainer import (
    NEARTrainer,
    NEARTrainerConfig,
)
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


class _ProgressBar(pl.callbacks.Callback):
    """
    callback that updates a progress bar once per epoch
    """

    def __init__(self, num_epochs, progress_bar):
        self.num_epochs = num_epochs
        self.progress_bar = progress_bar

    def on_train_epoch_end(self, trainer, pl_module):
        self.progress_bar.update(1)
        # set train and val loss in progress bar
        self.progress_bar.set_postfix(
            train_loss=trainer.callback_metrics["train_loss"].item(),
            val_loss=trainer.callback_metrics["val_loss"].item(),
        )


class ValidationCost:
    """
    A class that computes the validation cost of a program using a neural DSL.
    This is epsilon-admissible heuristic: https://arxiv.org/abs/2007.12101

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param error_loss: The loss to return if the program is invalid.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param callbacks: Callbacks to use during training.
    :param kwargs: Additional arguments to pass to the trainer.
    """

    def __init__(
        self,
        *,
        neural_dsl: DSL,
        trainer_cfg: NEARTrainerConfig,
        datamodule: DatasetWrapper,
        error_loss=10000,
        progress_by_epoch=False,
        callbacks: List[pl.callbacks.Callback] = (),
        **kwargs,
    ):
        self.neural_dsl = neural_dsl
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.error_loss = error_loss
        self.kwargs = kwargs
        self.progress_by_epoch = progress_by_epoch
        self.callbacks = list(callbacks)

    def __call__(self, node: DSLSearchNode) -> float:
        """
        Initialize a pl.Trainer object and train on a partial program.
        Returns validation cost after training.

        :param node: The partial program DSLSearchNode to compute the score for.

        :returns: The validation loss as a `float`.
        """
        trainer, pbar = self._get_trainer_and_pbar(
            label=render_s_expression(node.program)
        )
        try:
            initialized_p = self.neural_dsl.initialize(node.program)
        except PartialProgramNotFoundError:
            return self.error_loss

        model = self.neural_dsl.compute(initialized_p)
        if not isinstance(model, torch.nn.Module):
            del initialized_p
            model = TorchProgramModule(dsl=self.neural_dsl, program=node.program)
        self._fit_trainer(trainer, model, pbar)
        return trainer.callback_metrics["val_loss"].item()

    @staticmethod
    def _duplicate(callbacks):
        """
        Reinitialize all callbacks to avoid sharing state between different validation runs.
        """
        out = []
        for cb in callbacks:
            out.append(
                cb.__class__(
                    **{
                        k: getattr(cb, k)
                        for k in cb.__init__.__code__.co_varnames
                        if hasattr(cb, k)
                    }
                )
            )
        return out

    def _get_trainer_and_pbar(self, label=None):
        callbacks = list(self.callbacks)
        callbacks = self._duplicate(self.callbacks)
        if self.progress_by_epoch:
            log(f"Training {label}")
            pbar = tqdm.tqdm(total=self.trainer_cfg.n_epochs, desc="Training")
            callbacks.append(_ProgressBar(self.trainer_cfg.n_epochs, pbar))
        else:
            pbar = None

        trainer = pl.Trainer(
            max_epochs=self.trainer_cfg.n_epochs,
            devices="auto",
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            **self.kwargs,
            callbacks=callbacks,
        )
        return trainer, pbar

    def _fit_trainer(self, trainer, model, pbar):
        pl_model = NEARTrainer(model, config=self.trainer_cfg)
        trainer.fit(
            pl_model,
            self.datamodule.train_dataloader(),
            self.datamodule.val_dataloader(),
        )
        if self.progress_by_epoch:
            pbar.close()
