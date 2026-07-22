"""
math : make_tasks.py | Author : Sagnik Anupam.

Utility functions for loading tasks for the math domain. This domain was designed in the "Neurosymbolic Reasoning for Mathematical Domains" project and builds on the mathematical domain in the Lemma and ConPoLe papers.
"""

import os

from src.task_loaders import *
from data.math.grammar import *

import dreamcoder.domains.math.makeMathTasks as math_legacy

DOMAIN_NAME = "math"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")

# Experiment Type 1: Original Cognitive Tutor Problems Only (Referred to as Original Training Dataset and Original Testing Dataset as the two sets randomly split in a 70-30 ratio)

"""
TASKS = "cognitiveTutor"
"""

# Experiment Type 2: Golden Dataset in Training + Cognitive Tutor in Testing
"""
TASKS = "goldenDataset"
"""

# Experiment Type 3: Original Training Dataset Augmented by Golden Dataset in Training + Original Testing Dataset in Training
"""
TASKS = "trainingAugmentedGoldenDataset"
"""
# Experiment Type 4: Original Training Dataset Augmented by Index Problems
TASKS = "trainingWithIndex"

# Experiment Type 4: Original Training Dataset Augmented by Easier Versions of Problem DreamCoder Failed to Solve Initially

DEFAULT_TASKS_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS)

@TaskLoaderRegistry.register
class MathLoader(TaskDataLoader):
    name = "math"

    def load_tasks(self):

        train_tasks, test_tasks = math_legacy.loadMathDataset(
            task_dataset=TASKS,
            task_dataset_dir=DEFAULT_DATA_DIRECTORY,
            type_request="tstr",
        )

        return {TRAIN: train_tasks, TEST: test_tasks}
    

@TaskLanguageLoaderRegistry.register
class MathLanguageLoader(TaskDataLoader):
    name = "math"

    def load_task_language(self):
        return ({},{}) # No language for math tasks
    

