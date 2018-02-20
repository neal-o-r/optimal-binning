# Optimal Binning

Routine to choose the optimal number of equally spaced bins for use in a histogram. Two methods are used, one based on [Hogg](https://arxiv.org/abs/0807.4820) and the other on [Knuth](https://arxiv.org/abs/physics/0605197). This is packaged into a scikit transformer that takes a vector $x$ and a binary label $y$, and once fit will return, for a given $x$, the mean $y$ in the corresponding optimised bin.
