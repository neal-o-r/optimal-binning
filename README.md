# Optimal Binning

Routine to choose the optimal number of non-equally spaced bins for use in a histogram. Two methods are used, both based on [Hogg](https://arxiv.org/abs/0807.4820). This is packaged into a scikit transformer that takes a vector *x* and a binary label *y*, and once fit will return, for a given *x*, the mean *y* in the corresponding optimised bin.

There are three versions:
  * ```binning.py```: This works out the optimal number of **equally** spaced bins -- seems to work well.
  * ```recursive_binning.py```: Start with a given number of equally spaced bins, recursively merge adjacent bins so long as the log likelihood increases.
  * ```double_recursive_binning.py```: Start with 1 bin and add bins left to right so long as it increases the likelihood, once these have been added the use the recursive binning above to prune back. This seems to work quite well on the limited number of test cases that I've looked at.
