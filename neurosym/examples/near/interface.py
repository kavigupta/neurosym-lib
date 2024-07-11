from typing import Callable

import numpy as np
import torch
from frozendict import frozendict
from sklearn.exceptions import NotFittedError

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.methods.near_example_trainer import (
    NEARTrainerConfig,
    classification_mse_loss,
)
from neurosym.examples.near.neural_dsl import NeuralDSL
from neurosym.examples.near.search_graph import near_graph
from neurosym.examples.near.validation import ValidationCost
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.types.type_string_repr import TypeDefiner, parse_type
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


class NEAR:
    """
    A scikit-learn like interface to interact with NEAR.

    This isn't a 'true' sklearn program because we use Pytorch Lightning underneath the hood.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_depth: int,
        lr: float = 1e-4,
        max_seq_len: int = 100,
        n_epochs: int = 10,
        accelerator: str = "cpu",
    ):
        """
        Instantiate the NEAR interface.

        :param input_dim: Dimensionality of the input features.
        :param output_dim: Dimensionality of the output predictions.
        :param max_depth: Maximum depth of the search graph.
        :param lr: Learning rate.
        :param max_seq_len: Maximum sequence length for modelling trajectories.
        :param n_epochs: Number of epochs for training.
        :param accelerator: Accelerator to use for training ('cpu' / 'cuda' / etc.).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.n_epochs = n_epochs
        self.max_depth = max_depth
        self.accelerator = accelerator
        self.dsl = None
        self.type_env = None
        self.neural_dsl = None
        self.loss_callback = None
        self.search_strategy = None

        self._is_fitted = False
        self._is_registered = False
        self.programs = None
        self.validation_params = None

    def register_search_params(
        self,
        dsl: DSL,
        type_env: TypeDefiner,
        neural_modules: dict,
        search_strategy: Callable,
        loss_callback: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = classification_mse_loss,
        validation_params: dict = frozendict(),
    ):
        """
        Registers the parameters for the program search.

        :param dsl: The domain-specific language.
        :param type_env: Type environment.
        :param neural_modules: Neural modules to fill holes in partial programs.
        :param search_strategy: A search strategy supported in `neurosym.search`
        :param loss_callback: Callable for the loss function used during training.
            Defaults to classification MSE loss.
        """
        self.dsl = dsl
        self.type_env = type_env
        self.neural_dsl = NeuralDSL.from_dsl(dsl=self.dsl, modules=neural_modules)
        self.search_strategy = search_strategy
        self.loss_callback = loss_callback
        self._is_registered = True
        self.validation_params = validation_params

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        program_signature: str,
        n_programs: int = 1,  # type: ignore
        validation_max_epochs: int = 2000,
    ):
        """
        Fits the NEAR model to the provided data.

        :param datamodule: Data module containing the training and validation data.
        :param program_signature: Type signature of the program to be synthesized.
        :param n_programs: Number of programs to synthesize.
        :return: A list of `n_programs` number of trained estimators.
        """
        sexprs = self._search(datamodule, program_signature, n_programs)

        self.programs = [
            self.train_program(sexpr, datamodule, max_epochs=validation_max_epochs)
            for (sexpr, cost) in sexprs
        ]

        return self.programs

    def _search(self, datamodule, program_signature, n_programs):
        if not self._is_registered:
            raise NameError(
                "Search Parameters not available. Call `register_search_params` first!"
            )

        validation_cost = self._get_validator(datamodule, **self.validation_params)

        g = near_graph(
            self.neural_dsl,
            parse_type(
                s=program_signature,
                env=self.type_env,
            ),
            is_goal=self.neural_dsl.program_has_no_holes,
            max_depth=self.max_depth,
        )

        iterator = self.search_strategy(g, validation_cost, max_depth=self.max_depth)

        sexprs = []
        try:
            while len(sexprs) < n_programs:
                node = next(iterator)
                self._is_fitted = True
                cost = validation_cost(node)
                sexprs.append((node.program, cost))

        except StopIteration as exc:
            if (not self._is_fitted) or (len(sexprs) == 0):
                raise StopIteration(
                    "No symbolic program found! Check logs and hyperparameters!"
                ) from exc

        sexprs = sorted(sexprs, key=lambda x: x[1])
        for i, (sexpr, cost) in enumerate(sexprs):
            log(f"({i}) Cost: {cost:.4f}, {render_s_expression(sexpr)}")
        return sexprs

    def _get_validator(self, datamodule, **kwargs):
        validation_params = dict(
            trainer_cfg=self._trainer_config(datamodule),
            neural_dsl=self.neural_dsl,
            datamodule=datamodule,
            enable_model_summary=False,
            progress_by_epoch=False,
            accelerator=self.accelerator,
        )
        validation_params.update(**kwargs)

        validation_cost = ValidationCost(**validation_params)
        return validation_cost

    def train_program(
        self,
        program: SExpression,
        datamodule: pl.LightningDataModule,  # type: ignore
        max_epochs: int,
        **kwargs,
    ):
        """
        Trains a program on the provided data.

        :param program: The symbolic expression representing the program to train.
        :param datamodule: Data module containing the training and validation data.
        :return: Trained TorchProgramModule.
        """
        log(f"Validating {render_s_expression(program)}")
        trainer_params = dict(self.validation_params.items())
        trainer_params.update(**kwargs, max_epochs=max_epochs)
        module, _ = self._get_validator(datamodule, **trainer_params).validate_model(
            program
        )
        return module

    def predict(self, X: np.ndarray):
        """
        Makes predictions using the fitted programs.

        :param X: Input data as a NumPy array.
        :return: List of predictions from each fitted program.
        """
        if not self._is_fitted:
            raise NotFittedError(
                "No fitted program found! Call 'fit' with appropriate arguments before using this program."
            )

        with torch.no_grad():
            pred = [program(torch.tensor(X)).cpu().numpy() for program in self.programs]
        return pred

    def _trainer_config(self, datamodule: pl.LightningDataModule) -> NEARTrainerConfig:
        return NEARTrainerConfig(
            lr=self.lr,
            max_seq_len=self.max_seq_len,
            n_epochs=self.n_epochs,
            num_labels=self.output_dim,
            train_steps=len(datamodule.train),
            loss_callback=self.loss_callback,
        )
