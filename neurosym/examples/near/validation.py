import copy

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    roc_curve,
)

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import (
    IdentityProgramEmbedding,
    MinimalStepsNearStructuralCost,
    NearCost,
    NearValidationHeuristic,
    ProgramEmbedding,
)
from neurosym.examples.near.methods.base_trainer import schedule_optimizer
from neurosym.examples.near.methods.near_example_trainer import NEARTrainerConfig
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


def _is_binary_indicator(arr: np.ndarray) -> bool:
    vals = np.unique(arr)
    return np.all(np.isin(vals, [0, 1]))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _looks_like_probabilities(arr: np.ndarray) -> bool:
    return np.min(arr) >= 0.0 and np.max(arr) <= 1.0


def _threshold_single_label(
    y_true_1d: np.ndarray, probs_1d: np.ndarray, threshold_type: str = "quantile"
) -> np.ndarray:
    uniq = np.unique(y_true_1d)
    if uniq.size < 2:
        return np.full_like(probs_1d, int(uniq[0]), dtype=np.int32)

    if threshold_type == "quantile":
        prevalence = float(y_true_1d.mean())
        if prevalence <= 0.0:
            return np.zeros_like(probs_1d, dtype=np.int32)
        if prevalence >= 1.0:
            return np.ones_like(probs_1d, dtype=np.int32)
        thr = np.quantile(probs_1d, 1.0 - prevalence)
    elif threshold_type == "roc":
        fpr, tpr, thresholds = roc_curve(y_true_1d, probs_1d)
        thr = thresholds[np.argmax(tpr - fpr)]
    elif threshold_type == "static":
        thr = 0.5
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    return (probs_1d >= thr).astype(np.int32)


def _prepare_targets_and_preds(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold_type: str = "quantile",
):
    """
    Normalize a wide variety of shapes into:
    - task = 'multilabel': y_true (N,L) in {0,1}, y_pred_bin (N,L) in {0,1}
    - task = 'multiclass': y_true (M,) ints, y_pred_cls (M,) ints
      (where M is N, or N*T for sequence tasks)
    """
    y_true = ground_truth
    y_pred = predictions

    # Case A: class axis present in predictions -> multiclass (e.g., (N,2) or (N,T,4))
    has_class_axis = (
        y_pred.ndim >= 2
        and y_pred.shape[-1] > 1
        and (
            y_true.ndim == y_pred.ndim - 1
            or (y_true.ndim == y_pred.ndim and y_true.shape[-1] == y_pred.shape[-1])
            or (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[1] == 1)
        )
    )

    if has_class_axis:
        probs = (
            _softmax(y_pred, axis=-1)
            if not _looks_like_probabilities(y_pred)
            and np.allclose(y_pred.sum(axis=-1), 1.0, atol=1e-3)
            else y_pred
        )
        y_pred_cls = probs.argmax(axis=-1)

        # Normalize ground truth to class indices
        if y_true.ndim == y_pred.ndim and y_true.shape[-1] == y_pred.shape[-1]:
            # One-hot or soft targets -> take argmax
            y_true_cls = y_true.argmax(axis=-1)
        elif y_true.ndim == y_pred.ndim - 1:
            y_true_cls = y_true
        elif y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[1] == 1:
            y_true_cls = y_true.squeeze(1)
        else:
            raise ValueError(
                f"Shape mismatch for multiclass: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        # Flatten any extra axes for metrics
        return (
            y_true_cls.reshape(-1).astype(int),
            y_pred_cls.reshape(-1).astype(int),
            "multiclass",
        )

    # Case B: multilabel (no class axis, same shape, binary indicators in y_true)
    same_shape = y_true.shape == y_pred.shape
    if same_shape and _is_binary_indicator(y_true):
        # Convert scores to probabilities if needed

        probs = _sigmoid(y_pred) if not _looks_like_probabilities(y_pred) else y_pred

        # Per-label thresholding
        if probs.ndim == 1:
            preds_bin = _threshold_single_label(
                y_true.astype(int), probs, threshold_type
            )
            y_true_ml = y_true.astype(int).reshape(-1)
            preds_bin = preds_bin.reshape(-1)
        elif probs.ndim == 2:
            cols = []
            for j in range(probs.shape[1]):
                cols.append(
                    _threshold_single_label(
                        y_true[:, j].astype(int), probs[:, j], threshold_type
                    )
                )
            preds_bin = np.stack(cols, axis=1).astype(np.int32)
            y_true_ml = y_true.astype(int)
        else:
            # Allow (N,1) after reshape
            preds_bin = _threshold_single_label(
                y_true.reshape(-1).astype(int), probs.reshape(-1), threshold_type
            )
            y_true_ml = y_true.reshape(-1).astype(int)
            preds_bin = preds_bin.reshape(-1)

        return y_true_ml, preds_bin, "multilabel"

    # Case C: binary special cases like (N,1) vs (N,2)
    if y_pred.ndim == 2 and y_pred.shape[1] == 2 and y_true.ndim in (1, 2):
        # Treat as multiclass with 2 classes
        probs = (
            y_pred
            if (
                _looks_like_probabilities(y_pred)
                and np.allclose(y_pred.sum(axis=-1), 1.0, atol=1e-3)
            )
            else _softmax(y_pred, axis=-1)
        )

        y_pred_cls = probs.argmax(axis=-1)
        y_true_cls = y_true.squeeze(-1) if y_true.ndim == 2 else y_true
        return (
            y_true_cls.reshape(-1).astype(int),
            y_pred_cls.reshape(-1).astype(int),
            "multiclass",
        )

    raise ValueError(
        f"Unsupported shapes. I expected either (N,) or (N,L) multilabel, "
        f"or ...xC multiclass with a class axis. Got y_true={y_true.shape}, y_pred={y_pred.shape}."
    )


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metric_name: str = "all",
    threshold_type: str = "quantile",
):
    """
    Generalized metrics:
      - Multilabel: threshold per label (quantile/roc/static)
      - Multiclass: softmax + argmax over class axis (supports sequence shapes)

    Returns dict with: f1_score (weighted), unweighted_f1 (macro), all_f1s (per class/label),
    hamming_accuracy (== accuracy for multiclass), and classification report for "all".
    """
    y_true_norm, y_pred_norm, task = _prepare_targets_and_preds(
        ground_truth, predictions, threshold_type
    )

    weighted_avg_f1 = unweighted_avg_f1 = 0.0
    all_f1 = []
    hamming_accuracy = 0.0

    try:
        if task == "multilabel":
            if metric_name in ("all", "weighted_f1"):
                weighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="weighted", zero_division=0
                )
            if metric_name in ("all", "unweighted_f1"):
                unweighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="macro", zero_division=0
                )
            if metric_name == "all":
                all_f1 = f1_score(
                    y_true_norm, y_pred_norm, average=None, zero_division=0
                )
            if metric_name in ("all", "hamming_accuracy"):
                hamming_accuracy = 1.0 - hamming_loss(y_true_norm, y_pred_norm)

            report = (
                classification_report(
                    y_true_norm, y_pred_norm, output_dict=True, zero_division=0
                )
                if metric_name == "all"
                else {}
            )

        else:  # multiclass
            if metric_name in ("all", "weighted_f1"):
                weighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="weighted", zero_division=0
                )
            if metric_name in ("all", "unweighted_f1"):
                unweighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="macro", zero_division=0
                )
            if metric_name == "all":
                all_f1 = f1_score(
                    y_true_norm, y_pred_norm, average=None, zero_division=0
                )

            # Use plain accuracy for multiclass; return it under the same key for API stability.
            if metric_name in ("all", "hamming_accuracy"):
                hamming_accuracy = accuracy_score(y_true_norm, y_pred_norm)

            report = (
                classification_report(
                    y_true_norm, y_pred_norm, output_dict=True, zero_division=0
                )
                if metric_name == "all"
                else {}
            )

    except ValueError as e:
        print(f"Error computing metrics: {e}")
        report = {}

    return {
        "f1_score": float(weighted_avg_f1),
        "unweighted_f1": float(unweighted_avg_f1),
        "all_f1s": np.asarray(all_f1).tolist() if len(np.shape(all_f1)) else all_f1,
        "hamming_accuracy": float(hamming_accuracy),
        "report": report,
        "task_type": task,
    }


class ValidationCost(NearValidationHeuristic):
    """
    A class that computes the validation cost of a program using a neural DSL.
    This is epsilon-admissible heuristic: https://arxiv.org/abs/2007.12101

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param structural_cost_weight: Linearly interpolates b/w structural cost and validation loss.
        The scale of the validation cost (float) and structural_cost (int) can
        vary so it's important to tune this for each new problem.
    :param kwargs: Additional arguments to pass to the trainer.
    """

    def __init__(
        self,
        *,
        trainer_cfg: NEARTrainerConfig,
        datamodule: DatasetWrapper,
        progress_by_epoch=False,
        n_epochs=None,
    ):
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.progress_by_epoch = progress_by_epoch
        self.n_epochs = n_epochs

    def with_n_epochs(self, n_epochs: int) -> "ValidationCost":
        """
        Returns a new ValidationCost object with a different number of epochs.
        """
        return ValidationCost(
            trainer_cfg=self.trainer_cfg,
            datamodule=self.datamodule,
            progress_by_epoch=self.progress_by_epoch,
            n_epochs=n_epochs,
        )

    def compute_cost(
        self, dsl: DSL, model: InitializedSExpression, embedding: ProgramEmbedding
    ) -> float:
        """
        Initializes a TorchProgramModule and trains it. Returns the trained module, and the
        validation loss.

        :param program: The program to validate.

        :returns: A tuple containing the trained TorchProgramModule and the validation loss.
        """
        log(f"Training {render_s_expression(model.uninitialize())}")

        model = self.program_to_module(dsl, model, embedding)

        val_loss = _train_model(
            model, self.datamodule, n_epochs=self.n_epochs, trainer_cfg=self.trainer_cfg
        )

        return val_loss

    def program_to_module(
        self, dsl: DSL, program: InitializedSExpression, embedding: ProgramEmbedding
    ) -> torch.nn.Module:
        """
        Convert a program to a TorchProgramModule, which can then be trained.
        This can be overriden in subclasses to provide custom behavior, e.g.,
        integrating the program module into a larger model.

        :param program: The program to convert.
        :returns: The full Torch nn.Module to train.
        """
        program_module = TorchProgramModule(dsl, program)

        model = embedding.embed_initialized_program(program_module)
        return model


def default_near_cost(
    *,
    trainer_cfg: NEARTrainerConfig,
    datamodule: DatasetWrapper,
    progress_by_epoch=False,
    embedding: ProgramEmbedding = IdentityProgramEmbedding(),
    structural_cost_weight: float = 0.5,
    symbol_costs=None,
    cost=MinimalStepsNearStructuralCost,
    **kwargs,
):
    """
    Default NearCost. This is, by default, a 50/50 blend of structural cost and validation cost,
    with the given parameters.

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param embedding: The embedding to use.
    :param structural_cost_weight: Linearly interpolates b/w structural cost and validation loss.
    :param kwargs: Additional arguments to pass to the trainer.
    """
    return NearCost(
        structural_cost=cost(symbol_costs=symbol_costs or {}),
        validation_heuristic=ValidationCost(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
            progress_by_epoch=progress_by_epoch,
            **kwargs,
        ),
        structural_cost_weight=structural_cost_weight,
        embedding=embedding,
    )


# pylint: disable=too-many-branches,too-many-statements
def _train_model(model, datamodule, *, n_epochs, trainer_cfg: NEARTrainerConfig):
    if n_epochs is None:
        n_epochs = trainer_cfg.n_epochs
    if any(
        p.requires_grad for p in model.parameters()
    ):  # only train if there are parameters to train
        torch.manual_seed(trainer_cfg.seed)
        best_val = float("inf")
        best_acc = 0.0
        epochs_no_improve = 0
        best_weights = None

        optimizer, schedulers = schedule_optimizer(
            trainer_cfg.optimizer(
                model.parameters(),
                lr=trainer_cfg.lr,
                weight_decay=trainer_cfg.weight_decay,
            ),
            trainer_cfg.scheduler,
            len(datamodule.train),
            n_epochs,
        )
        model = model.to(trainer_cfg.accelerator)
        for epoch in range(n_epochs):
            model.train()
            for batch in datamodule.train_dataloader():
                optimizer.zero_grad()
                batch = [v.to(trainer_cfg.accelerator) for v in batch]
                x, y = batch
                loss = trainer_cfg.loss_callback(model(x, environment=()), y)
                loss.backward()
                optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()

            if epoch == (n_epochs - 1) or epoch % trainer_cfg.validation_interval == 0:
                model.eval()
                val_loss_sum, val_cnt = 0.0, 0
                labels, predictions = [], []
                with torch.no_grad():
                    for batch in datamodule.val_dataloader():
                        batch = [v.to(trainer_cfg.accelerator) for v in batch]
                        x, y = batch
                        pred_y = model(x, environment=())
                        val_loss_sum += trainer_cfg.loss_callback(pred_y, y).item()
                        val_cnt += 1
                        labels.append(y.cpu())
                        predictions.append(pred_y.cpu())
                val_loss = val_loss_sum / val_cnt
                labels = torch.cat(labels, dim=0)
                predictions = torch.cat(predictions, dim=0).detach()
                metrics = compute_metrics(
                    predictions=predictions.cpu().numpy(),
                    ground_truth=labels.cpu().numpy(),
                    metric_name="all",
                )
                metric_key = trainer_cfg.validation_metric
                if metric_key not in metrics:
                    raise KeyError(
                        f"Validation metric '{metric_key}' is unavailable. "
                        f"Available metrics: {list(metrics.keys())}"
                    )
                val_acc = float(metrics[metric_key])
                best_acc = max(best_acc, val_acc)

                if trainer_cfg.early_stopping:
                    if val_loss < best_val - trainer_cfg.early_stopping_min_delta:
                        best_val = val_loss
                        best_weights = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= trainer_cfg.early_stopping_patience:
                            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.cpu()
    # return best_val
    return -1 * best_acc
