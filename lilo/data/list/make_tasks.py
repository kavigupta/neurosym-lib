"""
list : make_tasks.py | Author : Sagnik Anupam.

Utility functions for loading tasks for the list domain from DreamCoder.
"""

import os

from src.task_loaders import *

import dreamcoder.domains.list.main as list_legacy

DOMAIN_NAME = "list"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")

@TaskLoaderRegistry.register
class ListLoader(TaskDataLoader):
    name = "list"

    def load_tasks(self):

        all_tasks = list_legacy.retrieveJSONTasks(
            filename=os.path.join(DEFAULT_DATA_DIRECTORY, "list_tasks2.json"),
        )[:105]
        
        train_tasks = all_tasks[14:20][3:-1] #+ list_legacy.make_list_bootstrap_tasks()
        test_tasks = all_tasks[20:26]
        
        # tasks = list_legacy.make_list_bootstrap_tasks()
        # train_tasks = tasks
        # test_tasks = tasks

        return {TRAIN: train_tasks, TEST: test_tasks}
    

@TaskLanguageLoaderRegistry.register
class ListLanguageLoader(TaskDataLoader):
    name = "list"

    def load_task_language(self):
        return ({},{}) # No language for list tasks