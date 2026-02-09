
## Case Study: NEAR Replication with `neurosym-lib`

One of the goals of neurosym-lib is to provide a framework for implementing neurosymbolic methods efficiently. In this section, we show how the NEAR algorithm [link paper] can be implemented natively using neurosym-lib. Our implementation proves to be enough to support extensions of NEAR with almost no added effort. Furthermore, our implementation is designed to be modular enough that we can easily extend it to support extensions that otherwise were not possible with the original implementation.

__Organization.__ Section A will detail the design decision of the original NEAR implementation. Section B showcases the process of using neurosym-lib to implement NEAR. In Section C, we showcase the faithfulness of neurosym-lib's representation to the original CRIM-13 dataset. Finally, Section D will showcase places where neurosym-lib _outperforms_ the original NEAR implementation, and the many possible extensions that we were able to implement with minimal effort.


### A. Design Decisions of the Original NEAR Implementation

__Background.__ NEAR is a program synthesis algorithm that is geared towards accelerating the search for differentiable programs. In top-down program synthesis, the best-fit program is found by recursively expanding the program tree, starting from the root node. Generally, any "complete" tree -- one that contains no unexpanded nodes -- can be evaluated against a set of training examples to determine an empirical loss. However, if a DSL is too complex, the search space over the set of _partial programs_ can be too large to explore exhaustively. NEAR addresses this by introducing _neural relaxations_ as a way to approximate the utility of a partial program. Intuitively, each hole of a differentiable partial program can be filled with a type accurate neural network. This neural network, through the universal approximation theorem, can provide an epsilon-admissible heuristic that lower-bounds the empirical loss of the complete program. Then, an off the shelf search algorithm, such as A*, can be used to find the best-fit program by expanding the partial programs with the lowest neural relaxation value.

__Design Decisions.__ NEAR is implemented in Python and PyTorch.

TODO: Discuss with Kavi. What design decisions do we want to highlight here?

Some pain points:
-  The original implementation requires hard-coding each operation of the DSL for each type class. For every new type, each function needs to be manually inspected and updated.
- There was no generic type system IIRC. All possible expansions had to be hard-coded. Some functions supported multiple types, which made the expansions logic somewhat weird to parse if you do not know the internal DSL.
- The loss function could not be easily changed. The training objective, similarly, could not be changed.
- The programs were expressed as a list of nested classes and there was no way to easily inspect, extend, modify, or serialize the programs. This made it hard to integrate NEAR into other systems or to extend it with new features.
- We should ask Megan why NEAR sucked.


### B. Implementing NEAR with `neurosym-lib`

<!-- I'm honestly not sure how to write this section... Maybe I can present a pseudocode of the new algorithm. What other stuff did we need to add to make NEAR work (in the folder section) -->


### C. Faithfulness of neurosym-lib's Representation to CRIM-13 Dataset

__Overview__ The CRIM-13 dataset contains keypoint-trajectories for a pair of mice engaging in social behavior. Each trajectory frame has been annotated for different actions by behavior experts. The goal of a model is to predict the action of the mice at each frame, given the trajectory data. Specifically, our goal is to learn a program for classifying the mice action at each frame for a fixed-size trajectory. Each frame is represented by a 19-dimensional feature tensor: 4 for the (x, y) position of each mouse, and the remaining 15 features are derived from the position such as instantaneous velocity, acceleration, and distance between the two mice. These features are pre-selected by domain experts to be relevant for the task of action classification. We learn programs for two actions that can be identified in the dataset: "sniff" and "other" ("other" is a catch-all annotation used when no interesting behavior is observed). The dataset contains over 12404 training, 3077 validation, and 2953 test trajectories, each with 100 frames each.

Some modifications:
 - Dynamic thresholding: The original NEAR implementation imposed a decision threshold of 0.5 for the classification task. However, this threshold may not be optimal for all datasets. Instead, for both implementations, we set the threshold to be the quantile where the true positive rate is equal to the false positive rate. This allows us to adapt the threshold to the specific dataset and task, improving classification performance.


__Results.__ We compare the performance of the original NEAR implementation with our neurosym-lib implementation on the CRIM-13 dataset. The results show that both implementations achieve similar performance, with our implementation being slightly faster due to the optimizations in neurosym-lib. The results are summarized in the table below:

| Method (Dataset: CRIM-13)                     |   f1-score |        time | hamming accuracy |
|:----------------------------------------------|-----------:|------------:|-----------------:|
| Original Implementation (A-star+ NEAR)        |  0.471933  | 852.341     | 0.893698         |
| `neurosym-lib` Implementation (A-star + NEAR) |  0.471933  | __112.097__ | 0.893698         |
| Reported Program Accuracy (Fig 2).            |  0.471933  | -           | 0.893698         |




### D. Extensions and Improvements with neurosym-lib
 - Parallel A-star search.
 - Other Heuristics.
    - Without `GenericMLPRNNNeuralHoleFiller`
    - With `GenericMLPRNNNeuralHoleFiller`
    - Case study on how easy/hard is it to add a new function to the search algo and integrate it into the search.
    - How easy is it to search over a language with variables instead of a langauge with combinators?
        - Repimplement the mouse DSL with lambdas/variables.
