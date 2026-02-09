
# Neurosymbolic Library

The `neurosym` library is a Python package for neurosymbolic program synthesis. We aim to provide a set of tools for DSL design, program search, and program abstraction in a self-contained package, allowing researchers to reuse DSLs, datasets, and algorithmic components across different research projects.


> [!NOTE]
> **Under Development.** This library is still under development. Please check back later for more updates. Do not use this library for production or research purposes yet, it *will* change in the future.


## Installation

You can install the library using uv:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv sync
# For development work, run:
$ uv sync --all-groups
$ uv run pre-commit install
$ uv pip install -e .
```

## Running on real-world Examples

The `neurosym` library is demonstrated on several real-world datasets, showcasing its applicability in various domains. Each example below includes a brief motivation, a link to more detailed reproduction instructions, and the primary command to run the experiment.

### CRIM-13 (Mice) Experiment

**Motivation:** This experiment focuses on classifying specific behaviors in mice using the CRIM-13 dataset. The CRIM-13 dataset contains trajectories of two mice performing various behaviors, and the goal is to automatically classify these behaviors. Our analysis focuses on two behaviors -- whether the mice is sniffing an object or not, and whether the behavir is interesting to the scientists or not.

**More Details:** For detailed instructions on dataset preparation, running the experiment, and analyzing results, refer to the [CRIM-13 (Mice) Experiment](./notebooks/crim13_reproduction/README.md) section.

**Replicate Experiment:**
```bash
$ uv run python notebooks/crim13_reproduction/benchmark_crim13.py
```

### Fly-v-Fly (Fruit Fly) Experiment

**Motivation:** This experiment involves classifying interactions between fruit flies using the Fly-v-Fly dataset. The Fly-v-Fly dataset contains trajectories of two fruit flies performing various behaviors, and the goal is to automatically classify these behaviors.

**More Details:** For detailed instructions on dataset preparation, running the experiment, and analyzing results, refer to the [Fly-v-Fly (Fruit Fly) Experiment](./notebooks/flyvfly_reproduction/README.md) section.

**Replicate Experiment:**
```bash
$ uv run python notebooks/flyvfly_reproduction/benchmark_flyvfly.py
```

### Basketball Experiment

**Motivation:** This experiment aims to classify offensive versus defensive behaviors in basketball players from video data. The dataset contains trajectories of basketball players performing various behaviors, and the goal is to automatically classify whether the behavior is offensive or defensive.

**More Details:** For detailed instructions on dataset preparation, running the experiment, and analyzing results, refer to the [Basketball Experiment](./notebooks/bball_reproduction/README.md) section.

**Replicate Experiment:**
```bash
$ uv run python notebooks/bball_reproduction/benchmark_bball.py
```
