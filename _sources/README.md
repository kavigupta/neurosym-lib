# The `neurosym` Library

The [`neurosym` library](https://github.com/kavigupta/neurosym-lib/) is a Python package for neurosymbolic program synthesis. We aim to provide a set of tools for DSL design, program search, and program abstraction in a self-contained package.

For the 2024 Neurosymbolic Programming Summer School, the we provide a set of notebooks provide a hands-on component (30 min each) to complement the tutorials. We provide code as an initial walk-through of baseline methods, with optional exercises for open-ended exploration. The dataset we'll be working is synthetically generated data. 

## Setup

To install the library, you can use pip:

```bash
pip install neurosym
```

NOTE: For the tutorial, we will be working on Google Colab, so you don't need to install the library locally. You can simply run `!pip install neurosym` in the notebook which will run a shell command to install the library.

## Asking for Help

Feel free to reach out to Kavi Gupta or Atharva Sehgal. You can reach out to us in two ways:

- In Person: Flag us down any time during the conference!
- Via email: Reach out at `kavig@mit.edu` or `atharvas@utexas.edu`.



## [Notebook 1 - Classification](https://neurosymbolic-learning.github.io/near_demo_classification.html)

The goal of this notebook is to provide a walk-through of the neurosymbolic programming pipeline with a synthetic classification task.

- Part 1: Data Exploration
- Part 2: DSL Generation
- Part 3: Program Generation
- Part 4: Program Inspection

## [Notebook 2 - Regression](https://neurosymbolic-learning.github.io/near_demo_regression.html)

The goal of this notebook is to provide a walk-through of the neurosymbolic programming pipeline with a synthetic regression task.

- Part 1: Data Exploration
    - We're going to define a function `datagen()` and plot trajectories generated with datagen.
    - **Exercise**: Before reading through the code, look at the trajectory plot and hypothesize what the underlying function might be. Write down what mathematical operators (`sin`, `pow`, `exp`, etc.) would be useful to discover the underlying function.
- Part 2: DSL Generation
    - We're going to formalize our intuition by writing a DSL. Write code for the DSL.
    - **Exercise**: Modify the DSL with the mathematical operators we wrote down earlier. 
- Part 3: Program Generation
    - We're going to use Neural guided search (NEAR) to search for the best-fit program in the DSL.
- Part 4: Program Inspection
    - We will render the program found by NEAR and inspect it's performance. 
    - **Exercise**: Inspect the program found after search. Try different hyperparamters.

## [Notebook 3 - Abstraction Learning](https://neurosymbolic-learning.github.io/discrete_exercise_skeleton.html)

- Part 1: Defining a DSL
- Part 2: Finding Programs
- Part 3: Abstraction Learning