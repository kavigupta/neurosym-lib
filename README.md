
# Neurosymbolic Library

The `neurosym` library is a Python package for neurosymbolic program synthesis. We aim to provide a set of tools for DSL design, program search, and program abstraction in a self-contained package, allowing researchers to reuse DSLs, datasets, and algorithmic components across different research projects.


## Under Development

NOTE: this library is still under development. Please check back later for more updates. Do not use this library for production or research purposes yet, it *will* change in the future.


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
