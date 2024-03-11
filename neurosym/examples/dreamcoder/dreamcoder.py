from dataclasses import dataclass
from typing import List, Tuple, Type, TypeVar

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
    theta: ProgramDistributionFamily,
    dsl: DSL,
    beams: List[List[SExpression]],
) -> ProgramDistribution:
    # TODO implement
    pass


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
) -> RecognitionModel:
    # TODO figure out if fantasy tasks are sampled every time. If so, put them into an inner loop

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
