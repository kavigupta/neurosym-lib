import torch
import torch.nn as nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.examples.near.operations.lists import fold_torch, map_torch

import os
import numpy as np


def bounce_dsl():
   #First, you need to define a DSL for the bouncing ball problem. Start with the DSL from the regression problem.
    #is there anything you need to remove?
   pass

dsl = bounce_dsl()
print("Defined DSL")

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
dataset_factory = lambda train_seed: DatasetWrapper(
        DatasetFromNpy(
            "./demodata/bounce_example/train_ex_data.npy",
            "./demodata/bounce_example/train_ex_labels.npy",
            train_seed,
        ),
        DatasetFromNpy(
            "./demodata/bounce_example/test_ex_data.npy",
            "./demodata/bounce_example/test_ex_labels.npy",
            None,
        ),
        batch_size=200,
    )
datamodule = dataset_factory(42)
input_dim, output_dim = 4,4
print("Data has been loaded.")

print("Now, you have to add the code to actually search for a program using NEAR");

#The code below assumes you found some top 3 programs and stored them in the best_program_nodes variable.
best_program_nodes = sorted(best_program_nodes, key=lambda x: x[1])
for i, (node, cost) in enumerate(best_program_nodes):
    print("({i}) Cost: {cost:.4f}, {program}".format(i=i, program=ns.render_s_expression(node.program), cost=cost))

#The function below is set up to further fine tune the program, test it, and return a set of values produced by it.
def testProgram(best_program_node):
    module = near.TorchProgramModule(dsl=neural_dsl, program=best_program_node[0].program)    
    pl_model = near.NEARTrainer(module, config=trainer_cfg)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-6, patience=5, verbose=False, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=4000,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        #enable_model_summary=False,
        #enable_progress_bar=False,
        logger=False,
        callbacks=[],
    )

    trainer.fit(
        pl_model, datamodule.train_dataloader(), datamodule.val_dataloader()
    )
    T = 100
    path = np.zeros((T, 4))
    X = torch.tensor(np.array([0.21413583, 4.4062634,  3.4344807,  0.12440437]), dtype=torch.float32)
    for t in range(T):
        path[t,:] = X.detach().numpy()                
        Y = module(X.unsqueeze(0)).squeeze(0)
        X = Y
    return path



#We generate trajectories for the top 2 programs.
trajectory = testProgram(best_program_nodes[0])
trajectoryb = testProgram(best_program_nodes[1])


#And then the code below plots it to show how it compares to a trajectory in the training set.
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize


title = "Trajectories and their quadrants"

print(trajectory[:])
plt.scatter(trajectory[:, 0], trajectory[:, 1], marker='o')
        
plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color='gray')

plt.scatter(trajectoryb[:, 0], trajectoryb[:, 1], marker='o')
        
plt.plot(trajectoryb[:, 0], trajectoryb[:, 1], alpha=0.2, color='gray')

truth = datamodule.train.inputs[0,:,:]

print(truth[0,:])

plt.scatter(truth[:, 0], truth[:, 1], marker='o')
        
plt.plot(truth[:, 0], truth[:, 1], alpha=0.2, color='orange')



plt.title(title)
plt.xlim(-5, 10)
plt.ylim(-5, 7)
plt.grid(True)
plt.show()

