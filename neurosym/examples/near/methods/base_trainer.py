import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from neurosym.utils.documentation import internal_only
from neurosym.utils.imports import import_pytorch_lightning

pl = import_pytorch_lightning()


@dataclass
class BaseTrainerConfig:
    """
    Base configuration for a trainer. See :py:class:`neurosym.examples.near.NEARTrainerConfig`
    for more details.
    """

    lr: float = 1e-4
    weight_decay: float = 0.0
    n_epochs: int = 10
    train_steps: int = -1  # Set programmaticaly
    seed: int = 44
    evaluate: bool = True
    resume: str = ""
    scheduler: str = "cosine"
    sav_dir: str = "data/checkpoints"
    _filter_param_list: Tuple[str] = ()
    optimizer: str = torch.optim.Adam


class BaseTrainer(pl.LightningModule):
    """
    An abstract class that defines the supporting code to
    train a neural module. We use pytorch-lightning as the
    base framework. This class exists primarily as the base
    class for :py:class:`neurosym.examples.near.NEARTrainer`.
    """

    @classmethod
    def from_arch(cls, model: nn.Module, config: BaseTrainerConfig):
        """
        Instantiates a neural module from a torch.nn.Module.
        """
        return cls(model, config)

    def __init__(self, model: nn.Module, config: BaseTrainerConfig):
        super().__init__()
        self.config = config
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    @internal_only
    def loss(self) -> dict:
        # pylint: disable=arguments-differ
        raise NotImplementedError("Loss function not implemented.")

    def _step(self) -> dict:
        raise NotImplementedError("Step function not implemented.")

    def training_step(self, train_batch, batch_idx):
        # pylint: disable=arguments-differ
        del batch_idx
        losses = self._step(validation=False, **train_batch)
        train_loss = sum(losses.values()) / len(losses)
        for key, value in losses.items():
            self.log(f"train_{key}", value.item(), prog_bar=True, sync_dist=True)

        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)
        return {"loss": train_loss}

    def _evaluation_step(self, evaluation_batch, batch_idx, split):
        # pylint: disable=arguments-differ
        del batch_idx
        losses = self._step(validation=True, **evaluation_batch)
        evaluation_loss = sum(losses.values()) / len(losses)
        for key, value in losses.items():
            self.log(f"{split}_{key}", value.item(), prog_bar=True, sync_dist=True)

        self.log(f"{split}_loss", evaluation_loss, prog_bar=True, sync_dist=True)
        return {f"{split}_loss": evaluation_loss}

    def validation_step(self, val_batch, batch_idx):
        # pylint: disable=arguments-differ
        return self._evaluation_step(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        # pylint: disable=arguments-differ
        return self._evaluation_step(test_batch, batch_idx, "test")

    @staticmethod
    def _filter_parameters(named_parameters: dict, filter_param_list: list):
        params = []
        for name, param in named_parameters:
            valid_name = not any(x in name for x in filter_param_list)
            if param.requires_grad and valid_name:
                params.append(param)
            else:
                warnings.warn(f"WARNING: Parameter {name} removed from optimizer")
        return params

    @staticmethod
    def _warm_and_decay_lr_scheduler(warmup_steps_pct, decay_steps_pct, total_steps):
        def f(step):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < (
                total_steps + 1
            ), f"Step {step} is greater thantotal steps {total_steps}"
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= 0.5 ** (step / decay_steps)
            return factor

        return f

    def _step_lr_scheduler(self, total_steps: int):
        def f(step: int):
            if step < total_steps * 0.3:
                factor = 1
            elif step < total_steps * 0.3:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        return f

    @internal_only  # not actually, it's just to silence the error that this isn't documented. It is an inherited method.
    def configure_optimizers(self):
        """
        A rather verbose function that instantiates the optimizer and scheduler.
        """
        # pylint: disable=protected-access
        params = self._filter_parameters(
            self.named_parameters(), self.config._filter_param_list
        )
        optimizer_fn = getattr(torch.optim, self.config.optimizer.__name__)
        optimizer = optimizer_fn(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        assert self.config.train_steps != -1, "Train steps not set"
        total_steps = int(self.config.n_epochs * (self.config.train_steps))

        match self.config.scheduler:
            case "none":
                return optimizer
            case "cosine":
                warmup_steps_pct = 0.02
                decay_steps_pct = 0.2
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=self._warm_and_decay_lr_scheduler(
                        warmup_steps_pct, decay_steps_pct, total_steps
                    ),
                )
                return (
                    [optimizer],
                    [
                        {
                            "scheduler": scheduler,
                            "interval": "step",
                        }
                    ],
                )
            case "step":
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer, lr_lambda=self._step_lr_scheduler(total_steps)
                )
                return (
                    [optimizer],
                    [
                        {
                            "scheduler": scheduler,
                            "interval": "step",
                        }
                    ],
                )
            case _:
                raise ValueError(f"Invalid scheduler {self.config.scheduler}")
