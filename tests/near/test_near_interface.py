# pylint: disable=R0801
import unittest
from functools import partial

import pytest

import neurosym as ns
from neurosym.examples import near


class TestNEARInterface(unittest.TestCase):
    def test_sequential_dsl_astar_interface(self):
        """
        Test sequential_dsl with Astar in the NEARInterface
        """
        datamodule = ns.datasets.near_data_example(train_seed=0)
        input_dim, output_dim = datamodule.train.get_io_dims()
        original_dsl = near.example_rnn_dsl(input_dim, output_dim)

        interface = near.NEAR(
            input_dim=input_dim,
            output_dim=output_dim,
            max_seq_len=100,
            n_epochs=10,
            max_depth=3,
        )

        t = ns.TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")
        interface.register_search_params(
            dsl=original_dsl,
            type_env=t,
            neural_modules={
                **near.create_modules(
                    "mlp",
                    [t("($fL) -> $fL"), t("($fL) -> $fO")],
                    near.mlp_factory(hidden_size=10),
                ),
                **near.create_modules(
                    "rnn_seq2seq",
                    [t("([$fL]) -> [$fL]"), t("([$fL]) -> [$fO]")],
                    near.rnn_factory_seq2seq(hidden_size=10),
                ),
                **near.create_modules(
                    "rnn_seq2class",
                    [t("([$fL]) -> $fL"), t("([$fL]) -> $fO")],
                    near.rnn_factory_seq2class(hidden_size=10),
                ),
            },
            search_strategy=partial(ns.search.bounded_astar, max_depth=3),
        )
        with pytest.raises(StopIteration):
            interface.fit(
                datamodule=datamodule,
                program_signature="([{f, $L}]) -> [{f, $O}]",
                n_programs=1,
            )
