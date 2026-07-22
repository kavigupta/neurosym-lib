import os
import json
from dreamcoder.type import *
from dreamcoder.task import Task

def loadMathDataset(task_dataset, task_dataset_dir, type_request: str ="tstr"):
    '''
    Loads tasks from the math domain.
    
    Args:
        task_dataset_dir (str): os.path to the directory with task datasets
        task_dataset (str): name of a directory in task_dataset_dir with a 
            test and train subdirectory with each containing a tasks.json
        type_request (str): defines the type of the returned task. Currently    
            math domain only handles tstr -> tstr (for tasks to convert equation in prefix notation and string type to a solution state in prefix notation and string type) and tint -> tint (for tasks to generate larger numbers to be used for indices by adding/multiplying integers)

    Returns:
        tuple: a tuple with the following format:
            (<list of Task objects that are training tasks>, <list of Task objects that are testing tasks>)
    '''

    dataset_path = os.path.join(task_dataset_dir, task_dataset)
    tasks = {"train": [], "test": []}

    if type_request=="tstr":
        request = arrow(tstr, tstr);
    elif type_request=="tint":
        request = arrow(tint, tint);
    else:
        print("Cannot make math domain tasks of type " + type_request)
        assert False
    for split in ("train", "test"):
        split_path = os.path.join(dataset_path, split)
        with open(os.path.join(split_path, "tasks.json")) as f:
            task_data = json.load(f)
            tasks[split] = [
                Task(
                    name=task["name"],
                    request=request,
                    examples= [((ex["i"],), ex["o"]) for ex in task["examples"]],
                    features=None,
                    cache=False,
                )
                for task in task_data
            ]
    return tasks["train"], tasks["test"]