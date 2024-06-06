import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import os
import numpy as np


inputs = np.load('./demodata/bounce_example/train_ex_data.npy')


X = inputs 

B, T, _ = X.shape


title = "Bouncing ball trajectories"

#Just plots the first 5 trajectories in the training data.    
for b in range(5):
    trajectory = X[b]
    
    plt.scatter(trajectory[:, 0], trajectory[:, 1], marker='o')
        
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color='gray')
    print(trajectory[:])

plt.title(title)
plt.xlim(-5, 10)
plt.ylim(-5, 7)
plt.grid(True)
plt.show()
