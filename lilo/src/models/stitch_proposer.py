"""
stitch_proposer.py | Author : Catherine Wong.

Library learning model that uses the Stitch compressor to propose libraries.
Expects an experiment_state with a GRAMMAR and FRONTIERs.
Updates GRAMMAR based on Stitch compression.
"""

import json
import logging
from collections import defaultdict

import stitch_core as stitch

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Invented, Program
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.laps_grammar import LAPSGrammar
from src.models.stitch_base import StitchBase
from src.task_loaders import ALL

LibraryLearnerRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LIBRARY_LEARNER
]


@LibraryLearnerRegistry.register
class StitchProposerLibraryLearner(StitchBase, model_loaders.ModelLoader):

    name = "stitch_proposer"

    compress_input_filename = "stitch_compress_input.json"
    compress_output_filename = "stitch_compress_output.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProposerLibraryLearner(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_compressed_grammar_mdl_prior_rank(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        include_samples,
        use_mdl_program: bool = False,
        beta_reduce_programs: bool = True,
        update_grammar: bool = True,
        replace_existing_abstractions: bool = True,
        **kwargs,
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) based on the training data description length to rerank the libraries.

        params:
            `use_mdl_program`: If True, compresses the single MDL program for each frontier.
                If False, compresses all programs in the frontier.
            `beta_reduce_programs`: Whether to beta reduce programs before compression.
                This will rewrite the programs into the base DSL, removing any abstractions.
            `update_grammar`: If True, updates the grammar in the experiment_state
                with the new inventions from compression. If False, runs compression
                and writes an inventions file, but leaves the grammar unaltered.
            `replace_existing_abstractions`: If True, replaces all existing abstractions
                with new abstractions after compression.
        """
        # NOTE(gg): Restrict to single split, otherwise working with rewritten frontiers is tricky
        assert len(task_splits) == 1
        split = task_splits[0]
        # split = "_".join(task_splits)

        # Update the grammar to remove all existing abstractions.
        if update_grammar and replace_existing_abstractions:
            experiment_state.models[model_loaders.GRAMMAR] = LAPSGrammar.fromGrammar(
                experiment_state.models[model_loaders.GRAMMAR], remove_abstractions=True
            )

        # Write frontiers for stitch.
        frontiers_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.compress_input_filename,
            split=split,
        )
        self.write_frontiers_to_file(
            experiment_state,
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            frontiers_filepath=frontiers_filepath,
            use_mdl_program=use_mdl_program,
            beta_reduce_programs=beta_reduce_programs,
            include_samples=include_samples,
        )

        # Call stitch compressor.
        abstractions, task_to_programs, programs_rewritten, tasks = self._compress(
            experiment_state,
            frontiers_filepath,
            split,
            max_arity=kwargs["max_arity"],
            iterations=kwargs["iterations"],
        )

        # Rebuild the set of frontiers
        # NOTE(GG): Messy logic due to `include_samples=True` returns all samples; consider refactor.
        frontiers_rewritten, sample_frontiers_rewritten = [], []
        for task_id, program in zip(tasks, programs_rewritten):
            matching_tasks = experiment_state.get_tasks_for_ids(
                task_splits[0],
                [task_id],
                include_samples=False,
                include_ground_truth_tasks=True,
            )
            if matching_tasks:
                if len(matching_tasks) > 1:
                    logging.warning(
                        f"Found multiple ({len(matching_tasks)}) tasks associated with task_id {task_id}"
                    )
                    rng = experiment_state.metadata[RANDOM_GENERATOR]
                    task = rng.choice(matching_tasks)
                else:
                    task = matching_tasks[0]

                frontier = Frontier(
                    frontier=[
                        FrontierEntry(
                            program=Program.parse(program),
                            logPrior=0.0,
                            logLikelihood=0.0,
                        )
                    ],
                    task=task,
                )
                frontiers_rewritten.append(frontier)
            else:
                matching_sample_tasks = experiment_state.get_tasks_for_ids(
                    task_splits[0],
                    [],
                    include_samples=True,
                    include_ground_truth_tasks=False,
                )
                if len(matching_sample_tasks) == 0:
                    raise ValueError(
                        f"Failed to find task associated with task_id {task_id}"
                    )

                task = matching_sample_tasks[0]
                frontier = Frontier(
                    frontier=[
                        FrontierEntry(
                            program=Program.parse(program),
                            logPrior=0.0,
                            logLikelihood=0.0,
                        )
                    ],
                    task=task,
                )
                sample_frontiers_rewritten.append(frontier)

        # Update the grammar with the new inventions.
        if update_grammar:
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            new_productions = [(0.0, p.infer(), p) for p in abstractions]
            new_grammar = LAPSGrammar(
                logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
                productions=grammar.productions + new_productions,
                continuationType=grammar.continuationType,
                initialize_parameters_from_grammar=grammar,
            )

            # Recompute production probabilities in grammar
            # TODO(GG): Recompute w/r/t sample frontiers as well?
            new_grammar = new_grammar.insideOutside(
                frontiers_rewritten, pseudoCounts=30, iterations=1
            )
            new_grammar = LAPSGrammar.fromGrammar(new_grammar)  # Wrap in LAPSGrammar

            experiment_state.models[model_loaders.GRAMMAR] = new_grammar

            print(
                f"Updated grammar (productions={len(grammar.productions)}) with {len(new_productions)} new abstractions."
            )

        # Rescore frontiers under grammar
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        frontiers_rewritten = [grammar.rescoreFrontier(f) for f in frontiers_rewritten]

        # Clear old frontiers and replace with rewritten
        experiment_state.reset_task_frontiers(task_split=split, task_ids=ALL)
        assert all([t.empty for t in experiment_state.task_frontiers[split].values()])
        experiment_state.update_frontiers(
            new_frontiers=frontiers_rewritten,
            maximum_frontier=grammar.maximum_frontier,
            task_split=split,
            is_sample=False,
        )

        # NOTE(GG): For now, do not replace sample frontiers, since we clear these out every iteration
        experiment_state.reset_samples(task_split=split)

    def _compress(
        self,
        experiment_state,
        frontiers_filepath,
        split,
        max_arity,
        iterations,
    ):
        with open(frontiers_filepath, "r") as f:
            frontiers_dict = json.load(f)
            stitch_kwargs = stitch.from_dreamcoder(frontiers_dict)

        stitch_kwargs.update(dict(eta_long=True, utility_by_rewrite=True))
        
        compression_result = stitch.compress(
        **stitch_kwargs,
        iterations=iterations,
        max_arity=max_arity,
        no_other_util=True,
        )
        abstractions = [
            Invented.parse(abs["dreamcoder"])
            for abs in compression_result.json["abstractions"]
        ]

        task_to_programs = defaultdict(list)
        for rewritten, task in zip(
            compression_result.json["rewritten_dreamcoder"], stitch_kwargs["tasks"]
        ):
            task_to_programs[task].append(rewritten)

        abstractions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.compress_output_filename,
            split=split,
        )
        with open(abstractions_filepath, "w") as f:
            json.dump(compression_result.json, f, indent=4)

        return (
            abstractions,
            task_to_programs,
            compression_result.json["rewritten_dreamcoder"],
            stitch_kwargs["tasks"],
        )
