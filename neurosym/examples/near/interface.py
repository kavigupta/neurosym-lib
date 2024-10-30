import itertools
from types import NoneType
from typing import Callable, Union

import torch
from frozendict import frozendict

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.methods.near_example_trainer import (
    NEARTrainerConfig,
    classification_mse_loss,
)
from neurosym.examples.near.neural_dsl import NeuralDSL
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
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
        n_epochs: int = 10,
        accelerator: str = "cpu",
    ):
        """
        Instantiate the NEAR interface.

        :param input_dim: Dimensionality of the input features.
        :param output_dim: Dimensionality of the output predictions.
        :param max_depth: Maximum depth of the search graph.
        :param lr: Learning rate.
        :param n_epochs: Number of epochs for training.
        :param accelerator: Accelerator to use for training ('cpu' / 'cuda' / etc.).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.max_depth = max_depth
        self.accelerator = accelerator
        self.dsl = None
        self.type_env = None
        self.neural_dsl = None
        self.loss_callback = None
        self.search_strategy = None

        self._is_registered = False
        self.validation_params = None

    def register_search_params(
        self,
        dsl: DSL,
        type_env: TypeDefiner,
        neural_hole_filler: NeuralHoleFiller,
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
        :param neural_hole_filler: Neural modules to fill holes in partial programs.
        :param search_strategy: A search strategy supported in `neurosym.search`
        :param loss_callback: Callable for the loss function used during training.
            Defaults to classification MSE loss.
        """
        self.dsl = dsl
        self.type_env = type_env
        self.neural_dsl = NeuralDSL.from_dsl(
            dsl=self.dsl, neural_hole_filler=neural_hole_filler
        )
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
        max_iterations: Union[int, NoneType] = None,
    ):
        """
        Fits the NEAR model to the provided data.

        :param datamodule: Data module containing the training and validation data.
        :param program_signature: Type signature of the program to be synthesized.
        :param n_programs: Number of programs to synthesize.
        :param validation_max_epochs: Maximum number of epochs for validation.
        :param max_iterations: Maximum number of iterations for the search.

        :return: A list of `n_programs` number of trained estimators.
        """
        sexprs = self._search(datamodule, program_signature, n_programs, max_iterations)

        return [
            self.train_program(sexpr, datamodule, max_epochs=validation_max_epochs)
            for sexpr in sexprs
        ]

    def _search(
        self,
        datamodule,
        program_signature,
        n_programs,
        max_iterations: Union[int, NoneType] = None,
    ):
        if not self._is_registered:
            raise NameError(
                "Search Parameters not available. Call `register_search_params` first!"
            )

        validation_cost = self._get_validator(datamodule, max_epochs=None)

        g = near_graph(
            self.neural_dsl,
            parse_type(
                s=program_signature,
                env=self.type_env,
            ),
            is_goal=lambda _: True,
            max_depth=self.max_depth,
            cost=validation_cost,
        )

        iterator = self.search_strategy(
            g, max_depth=self.max_depth, max_iterations=max_iterations
        )

        sexprs = list(itertools.islice((prog.program for prog in iterator), n_programs))
        return sexprs

    def _get_validator(self, datamodule, max_epochs):

        validation_cost = ValidationCost(
            trainer_cfg=self._trainer_config(max_epochs),
            neural_dsl=self.neural_dsl,
            datamodule=datamodule,
            accelerator=self.accelerator,
            **self.validation_params,
        )
        return validation_cost

    def train_program(
        self,
        program: SExpression,
        datamodule: pl.LightningDataModule,  # type: ignore
        max_epochs: int,
    ):
        """
        Trains a program on the provided data.

        :param program: The symbolic expression representing the program to train.
        :param datamodule: Data module containing the training and validation data.
        :return: Trained TorchProgramModule.
        """
        log(f"Validating {render_s_expression(program)}")
        module, _ = self._get_validator(
            datamodule, max_epochs=max_epochs
        ).validate_model(program)
        return module

    def _trainer_config(self, max_epochs) -> NEARTrainerConfig:
        return NEARTrainerConfig(
            lr=self.lr,
            n_epochs=(
                self.n_epochs if max_epochs is None else min(self.n_epochs, max_epochs)
            ),
            loss_callback=self.loss_callback,
        )
