import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


class RecursiveOptimalBin(BaseEstimator, TransformerMixin):
    """
        Take in a 1d array and a set of binary labels (x, y)
        and return the mean y in each of a set of
        'optimal' bins
        """

    def __init__(self, a=10, max_bins=100, method="pdf"):
        """
            Init with 3 args:
                    a        : smoothing factor
                    max_bins : most bins that will be tested
            """
        self.max_bins = max_bins
        self.a = a

    def _lnL(self, bins, x, y):
        """
            Log likelihood from Hogg 2008
            """
        N, e, _ = stats.binned_statistic(x, statistic="sum", values=y, bins=bins)

        d = np.diff(e)

        if any((N + self.a) < 1):
            return np.nan

        s = np.sum(N + self.a)
        L = np.sum(N * np.log((N + self.a - 1) / (d * (s - 1))))

        return L

    def optimal_binning(self, x, y):
        """
            Find the optimal binning for a data set
            Begin with 100 bins, and keep combining
            adjacent bins, so long as it increases the
            log-likelihood.
            """
        # make 100 even bins
        bins = list(np.histogram(x, bins=self.max_bins)[1])

        # get the log L for that, set up loop variables
        L = self._lnL(bins, x, y)
        L_new = L
        bins_new = bins
        while (L_new >= L) and (len(bins_new) > 2):
            # so long as the lgL is increasing, and we have
            # at least 2 bin edges (start and end)
            L = L_new
            bins = bins_new
            Ls = []
            for i in range(1, len(bins) - 1):
                test_bins = bins[:i] + bins[i + 1 :]
                Ls.append(self._lnL(test_bins, x, y))

            L_new = np.nanmax(Ls)
            L_ind = 1 + np.nanargmax(Ls)
            bins_new = bins[:L_ind] + bins[L_ind + 1 :]

        return bins

    def fit(self, x, y):
        """
            Get the optimal bin number and create those bins,
            then aggregate the labels accordingly and populate
            self.mu
            """
        self.bins = self.optimal_binning(x, y)

        pdf = lambda a: np.sum(a) / np.sum(y)
        self.mu = stats.binned_statistic(x, statistic=pdf, values=y, bins=self.bins)[0]
        return self

    def transform(self, x):
        """
            get the bin that each x fell into and return
            the mu in corresponding to that bin
            """
        locs = np.digitize(x, self.bins)
        # np hist and digitize handle bins differently
        # so this fudge is required (numpy/issues/9208)
        locs -= 1
        locs[locs == len(self.mu)] -= 1
        return self.mu[locs]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


if __name__ == "__main__":

    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(123)

    binner = RecursiveOptimalBin(max_bins=200)
    x = np.concatenate(
        [
            stats.cauchy(-5, 1.8).rvs(500),
            stats.cauchy(-4, 0.8).rvs(2000),
            stats.cauchy(-1, 0.3).rvs(500),
            stats.cauchy(2, 0.8).rvs(1000),
            stats.cauchy(4, 1.5).rvs(500),
        ]
    )

    # truncate values to a reasonable range
    x = x[(x > -15) & (x < 15)]

    y = np.ones_like(x)

    plt.hist(x, density=True, bins=100, histtype="stepfilled", alpha=0.2)
    binner.fit(x, y)

    plt.hist(x, density=True, bins=binner.bins, histtype="step")

    plt.show()
