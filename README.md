
# Neurosymbolic Library

The `neurosym` library is a Python package for neurosymbolic program synthesis. We aim to provide a set of tools for DSL design, program search, and program abstraction in a self-contained package, allowing researchers to reuse DSLs, datasets, and algorithmic components across different research projects.


## Under Development

NOTE: this library is still under development. Please check back later for more updates. Do not use this library for production or research purposes yet, it *will* change in the future.

## Executing DreamCoder
```
docker build -t ns .
docker run --name ns -it ns
python3 run_iterative_experiment.py  --experiment_name toy --experiment_type dreamcoder --domain toy --encoder toy --iterations 2 --global_batch_sizes 95 --enumeration_timeout 1000 --recognition_train_steps 10 --random_seeds 111 --verbose
```