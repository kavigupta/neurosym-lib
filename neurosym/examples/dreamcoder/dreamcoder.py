import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Type, TypeVar

import numpy as np
import torch
from more_itertools import chunked

from neurosym.compression.process_abstraction import single_step_compression
from neurosym.dsl.dsl import DSL
from neurosym.program_dist.distribution import (
    ProgramCountsTensorBatch,
    ProgramDistribution,
    ProgramDistributionBatch,
    ProgramDistributionFamily,
)
from neurosym.program_dist.tree_distribution.tree_dist_enumerator import (
    enumerate_tree_dist,
)
from neurosym.programs.s_expression import SExpression

Input, Output = TypeVar("Input"), TypeVar("Output")


@dataclass
class IOExample:
    inputs: List[Input]
    output: Output


@dataclass
class Task:
    io_examples: List[IOExample]


class RecognitionModel(torch.nn.Module, ABC):
    def __init__(self, family: ProgramDistributionFamily):
        super().__init__()
        self.family = family

    @abstractmethod
    def run(self, tasks: List[Task]) -> ProgramDistributionBatch:
        pass


class Domain(ABC):
    @abstractmethod
    def task_log_prob(self, program: SExpression, task: Task) -> float:
        pass

    @abstractmethod
    def sample_task(self, program: SExpression, rng: np.random.RandomState) -> Task:
        pass


def dreamcoder(
    dsl: DSL,
    domain: Domain,
    tasks: List[Task],
    program_distribution_family: ProgramDistributionFamily,
    recoginition_model_class: Type[RecognitionModel],
    rng: np.random.RandomState,
    batch_size: int,
    enumeration_timeout: int,
    beam_size: int,
):
    """
    Implementation of Algorithm 1 from the DreamCoder paper (page 43)

    https://www.cs.cornell.edu/~ellisk/documents/dreamcoder_with_supplement.pdf
    """
    theta: ProgramDistribution = (
        program_distribution_family.default_distribution()
    )  # TODO implement
    beams = [[] for _ in tasks]
    while True:
        task_shuf = list(range(len(tasks)))
        rng.shuffle(task_shuf)
        for batch in chunked(task_shuf, batch_size):
            for t in batch:
                beams[t] += enumerate_programs(
                    program_distribution_family,
                    dsl,
                    theta,
                    t.io_examples,
                    timeout=enumeration_timeout,
                )
            recognition_model, _ = train_recogintion_model(
                recoginition_model_class=recoginition_model_class,
                dsl=dsl,
                domain=domain,
                family=program_distribution_family,
                theta=theta,
                tasks=tasks,
                beams=beams,
                rng=rng,
                probability_dreaming=0.5,
                time_limit_s=60,
                batch_size=batch_size,
            )
            dists = recognition_model.run(tasks)
            for t in batch:
                beams[t] += enumerate_programs(
                    program_distribution_family,
                    dsl,
                    dists[t],
                    t.io_examples,
                    timeout=enumeration_timeout,
                )
            for t in batch:
                beams[t] = prune(dsl, domain, theta, tasks[t], beams[t], beam_size)
            dsl, beams = compress(dsl, beams)
            # TODO look into how dreamcoder fits this
            theta = fit(program_distribution_family, beams)
            yield (dsl, theta), recognition_model, beams


def prune(
    dsl: DSL,
    domain: Domain,
    theta: ProgramDistributionFamily,
    task: Task,
    programs: List[SExpression],
    beam_size: int,
) -> List[SExpression]:
    programs = sorted(
        programs,
        key=lambda p: theta.log_prob(dsl, p) + domain.task_log_prob(p, task),
    )
    return programs[:beam_size]


def compress(
    dsl: DSL, beams: List[List[SExpression]]
) -> Tuple[DSL, List[List[SExpression]]]:
    programs_flat, tasks_flat = [], []
    for t, beam in enumerate(beams):
        programs_flat += beam
        tasks_flat += [str(t)] * len(beam)
    dsl, rewritten_flat = single_step_compression(dsl, programs_flat, tasks=tasks_flat)
    rewritten = []
    i = 0
    for beam in beams:
        rewritten.append(rewritten_flat[i : i + len(beam)])
        i += len(beam)
    return dsl, rewritten


def fit(
    family: ProgramDistributionFamily,
    beams: List[List[SExpression]],
) -> ProgramDistribution:
    # TODO fix this to make it more similar to the algorithm in the paper
    # Should implement page 54 from https://www.cs.cornell.edu/~ellisk/documents/dreamcoder_with_supplement.pdf
    # Specifically section S4.5.4
    # Maddy: We left of at
    # The question is did they ever implement inside_outside for bigrams
    # ie was bigram NEVER used outside of the recognition model
    # ie was it never used for enumeration without a neural net
    # AND never used by the inside_outside bit of compression that is used to score abstractions
    # And the implication of this would be theyre using a fit unigram model to sample the tasks instead of a bigram
    # Maddy leans toward that being the case but we should figure it out
    # Kavi: stupid idea, but we could literally just train a "recoginition model" that's just a constant set of weights
    # If this is convex, then it should find the optimum.

    # For now, just doing something simple.

    weights = [1 / len(beam) for beam in beams for _ in beam]

    counts = family.count_programs(
        [program for beam in beams for program in beam], weights=weights
    )
    return family.counts_to_distribution(counts)


def train_recogintion_model(
    recoginition_model_class: Type[RecognitionModel],
    dsl: DSL,
    domain: Domain,
    family: ProgramDistributionFamily,
    theta: ProgramDistribution,
    tasks: List[Task],
    beams: List[List[SExpression]],
    rng: np.random.RandomState,
    probability_dreaming: float,
    time_limit_s: int,
    batch_size: int,
    lr: float = 1e-3,
) -> RecognitionModel:

    beam_counts = [
        [family.count_programs(program) for program in beam] for beam in beams
    ]

    task_log_probs = [
        domain.task_log_prob(program, t) for program, t in zip(beams, tasks)
    ]

    def data_epoch():
        for t, b, log_prob in zip(tasks, beam_counts, task_log_probs):
            if rng.rand() < probability_dreaming:
                # TODO allow this to happen in parallel in some producer-consumery way
                program = theta.sample_program(dsl, rng)
                task = domain.sample_task(program, rng)
                counts = [family.count_programs(program)]
                yield task, counts, log_prob
            else:
                yield t, b, log_prob

    def data_iterator():
        while True:
            yield from data_epoch()

    recognition_model = recoginition_model_class(family)
    optimizer = torch.optim.Adam(recognition_model.parameters(), lr=lr)
    start_time = time.time()
    last_print = start_time
    losses = []
    for batch in chunked(data_iterator(), batch_size):
        if time.time() - start_time > time_limit_s:
            break
        if time.time() - last_print > 60:
            print(f"Training recognition model: {losses[-1]}")
            last_print = time.time()
        batch_tasks, batch_beam_counts, batch_task_log_probs = zip(*batch)
        loss = training_loss(
            recognition_model,
            family,
            batch_tasks,
            batch_beam_counts,
            batch_task_log_probs,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return recognition_model, losses


def training_loss(
    model: RecognitionModel,
    family: ProgramDistributionFamily,
    tasks: List[Task],
    beam_counts: List[ProgramCountsTensorBatch],
    task_log_probs: List[float],
) -> float:
    tensors = model(tasks)
    log_likelihood = torch.sum(
        [
            torch.max(
                [family.log_likelihood(tensor, count) + task_log_prob for count in beam]
            )
            for tensor, beam, task_log_prob in zip(tensors, beam_counts, task_log_probs)
        ]
    )
    return -log_likelihood


def enumerate_programs(
    family: ProgramDistributionFamily,
    dsl: DSL,
    theta: ProgramDistribution,
    io_examples: List[IOExample],
    timeout: int,
) -> List[SExpression]:
    """
    Enumerate programs that are likely to be good on the given examples.
    """
    start_time = time.time()
    for program, _ in enumerate_tree_dist(family.tree_distribution(theta)):
        if time.time() - start_time > timeout:
            break
        if all(
            program.evaluate(dsl, example.inputs) == example.output
            for example in io_examples
        ):
            yield program
