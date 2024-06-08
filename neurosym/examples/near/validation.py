import torch

from neurosym.examples.near.methods.near_example_trainer import NEARTrainer
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.utils.imports import import_pytorch_lightning


class ValidationCost:
    def __init__(
        self, *, neural_dsl, trainer_cfg, datamodule, error_loss=10000, **kwargs
    ):
        self.neural_dsl = neural_dsl
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.error_loss = error_loss
        self.kwargs = kwargs

    def __call__(self, node):
        pl = import_pytorch_lightning()

        trainer = pl.Trainer(
            max_epochs=self.trainer_cfg.n_epochs,
            devices="auto",
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            **self.kwargs,
        )
        try:
            initialized_p = self.neural_dsl.initialize(node.program)
        except PartialProgramNotFoundError:
            return self.error_loss

        model = self.neural_dsl.compute(initialized_p)
        if not isinstance(model, torch.nn.Module):
            del model
            del initialized_p
            model = TorchProgramModule(dsl=self.neural_dsl, program=node.program)
        pl_model = NEARTrainer(model, config=self.trainer_cfg)
        trainer.fit(
            pl_model,
            self.datamodule.train_dataloader(),
            self.datamodule.val_dataloader(),
        )
        return trainer.callback_metrics["val_loss"].item()
