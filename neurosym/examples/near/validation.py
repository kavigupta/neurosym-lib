import torch

from neurosym.examples.near.methods.near_example_trainer import NEARTrainer
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.utils.imports import import_pytorch_lightning


def validation_cost(node, *, neural_dsl, trainer_cfg, datamodule, error_loss=10000, **kwargs):
    pl = import_pytorch_lightning()

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.n_epochs,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        **kwargs,
    )
    try:
        initialized_p = neural_dsl.initialize(node.program)
    except PartialProgramNotFoundError:
        return error_loss

    model = neural_dsl.compute(initialized_p)
    if not isinstance(model, torch.nn.Module):
        del model
        del initialized_p
        model = TorchProgramModule(dsl=neural_dsl, program=node.program)
    pl_model = NEARTrainer(model, config=trainer_cfg)
    trainer.fit(pl_model, datamodule.train_dataloader(), datamodule.val_dataloader())
    return trainer.callback_metrics["val_loss"].item()
