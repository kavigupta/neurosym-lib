"""
math: encoder.py | Author : Sagnik Anupam

Same format as encoder.py in compositional_graphics/Re2, but for the math domain.
"""

from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL

from dreamcoder.domains.re2.main import StringFeatureExtractor

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

@ExamplesEncoderRegistry.register
class MathFeatureExamplesEncoder(ModelLoader):
    """
    Loads the Re2 Feature Extractor class for applications in the math domain. Note that this does not return an initialized model. It returns the class that can be instantiated from the experiment state, with other params set.
    """

    name = "math"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)
            re2_punctuation = [".", "(", ")", "|", "^", "[", "]"]
            math = ["=", "/", "*", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " "]
            return StringFeatureExtractor(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                punctuation=re2_punctuation+math,
                **kwargs
            )

        return experiment_state_initializer
