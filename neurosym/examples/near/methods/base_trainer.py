import torch

from neurosym.utils.imports import import_pytorch_lightning

pl = import_pytorch_lightning()


def schedule_optimizer(optimizer, scheduler_type, train_steps, n_epochs):
    """
    Schedules the optimizer based on the scheduler_type.

    :param optimizer: The optimizer to schedule.
    :param scheduler_type: The type of scheduler to use.
    :param train_steps: The number of training steps.
    :param n_epochs: The number of epochs to train for.

    :return: The scheduled optimizer and a list of schedulers.
    """
    assert train_steps != -1, "Train steps not set"
    total_steps = int(n_epochs * train_steps)

    match scheduler_type:
        case "none":
            return optimizer, []
        case "cosine":
            warmup_steps_pct = 0.02
            decay_steps_pct = 0.2
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=_warm_and_decay_lr_scheduler(
                    warmup_steps_pct, decay_steps_pct, total_steps
                ),
            )
            return optimizer, [scheduler]
        case "step":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=_step_lr_scheduler(total_steps)
            )
            return optimizer, [scheduler]
        case _:
            raise ValueError(f"Invalid scheduler {scheduler_type}")


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


def _step_lr_scheduler(total_steps: int):
    def f(step: int):
        if step < total_steps * 0.3:
            factor = 1
        elif step < total_steps * 0.3:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    return f
