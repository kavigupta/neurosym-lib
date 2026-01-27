"""
list: encoder.py | Author : Sagnik Anupam

Same format as encoder.py in compositional_graphics/Re2, but for the list domain.
"""

from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL

from dreamcoder.domains.list.main import LearnedFeatureExtractor

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

@ExamplesEncoderRegistry.register
class ListFeatureExamplesEncoder(ModelLoader):
    """
    Loads the LearnedFeatureExtractor for the list domain.
    """

    name = "list"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)
            return LearnedFeatureExtractor(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                **kwargs
            )

        return experiment_state_initializer
