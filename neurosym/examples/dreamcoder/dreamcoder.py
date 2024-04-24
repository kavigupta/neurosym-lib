from dataclasses import dataclass
import time
from typing import List, Tuple, Type, TypeVar
from more_itertools import chunked

import numpy as np
import torch
from neurosym.dsl.dsl import DSL
from neurosym.program_dist.distribution import (
    ProgramDistribution,
    ProgramDistributionFamily,
    ProgramsCountTensor,
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
                beams[t] += program_distribution_family.enumerate_programs(
                    dsl, theta, t.io_examples, rng, timeout=enumeration_timeout
                )
            recognition_model = train_recogintion_model(
                recoginition_model_class, dsl, domain, theta, tasks, beams, rng
            )
            for t in batch:
                beams[t] += program_distribution_family.enumerate_programs(
                    dsl,
                    recognition_model(tasks[t]),
                    t.io_examples,
                    rng,
                    timeout=enumeration_timeout,
                )
            for t in batch:
                beams[t] = prune(dsl, domain, theta, tasks[t], beams[t], beam_size)
            dsl, beams = compress(dsl, beams)
            # TODO look into how dreamcoder fits this
            theta = fit(program_distribution_family, dsl, beams)
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
    # TODO implement. keep tasks separate
    pass


def fit(
    family: ProgramDistributionFamily,
    dsl: DSL,
    beams: List[List[SExpression]],
) -> ProgramDistribution:
    # TODO implement
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
    beam_counts: List[List[ProgramsCountTensor]],
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
