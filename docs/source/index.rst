Welcome to ``neurosym``'s documentation!
===========================================

The ``neurosym`` library is a Python package for neurosymbolic program synthesis. We aim to provide a set
of tools for DSL design, program search, and program abstraction in a self-contained package, allowing
researchers to reuse DSLs, datasets, and algorithmic components across different projects.

Some examples of what you can do with ``neurosym``:

- Use the NeAR algorithm on a custom-designed DSL to synthesize programs for a basic physics simulation
  (`tutorial <https://github.com/kavigupta/neurosym-lib/blob/main/tutorial/bouncing_ball_exercise_skeleton.ipynb>`__,
  `solutions <https://github.com/kavigupta/neurosym-lib/blob/main/tutorial/bouncing_ball_exercise_solutions.ipynb>`__)
- Enumerate and compress symbolic expressions in a custom DSL
  (`tutorial <https://github.com/kavigupta/neurosym-lib/blob/main/tutorial/discrete_exercise_skeleton.ipynb>`__,
  `solutions <https://github.com/kavigupta/neurosym-lib/blob/main/tutorial/discrete_exercise_solutions.ipynb>`__)

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   s_expressions
   type
   dsls
   dist
   search_graph
   datasets
   compression
   python
   examples
   utils
