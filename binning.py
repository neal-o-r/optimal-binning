import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


class OptimalBin(BaseEstimator, TransformerMixin):
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

    def _lnL(self, n_bins, x, y):
        """
                Log likelihood from Hogg 2008
                """
        N, e, _ = stats.binned_statistic(x, statistic="sum", values=y, bins=n_bins)
        d = abs(e[0] - e[1])

        if any((N + self.a) < 1):
            return np.nan

        s = np.sum(N + self.a)
        L = np.sum(N * np.log((N + self.a - 1) / (d * (s - 1))))

        return L

    def _optimal_bin_no(self, x, y):
        """
                Step through the bins in 2's and evaluate the log-like
                """
        bins = np.arange(2, self.max_bins, 2).astype(int)

        ls = []
        for b in bins:
            l = self._lnL(b, x, y)
            # if you get a nan you'll get nans for all
            # subsequent bins
            if np.isnan(l):
                break

            ls.append(l)
        return bins[np.argmax(ls)]

    def fit(self, x, y):
        """
                Get the optimal bin number and create those bins,
                then aggregate the labels accordingly and populate
                self.mu
                """
        self.bin_no = self._optimal_bin_no(x, y)
        self.bins = np.histogram(x, bins=self.bin_no)[1]

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
    import numpy as np; np.random.seed(123)

    binner = OptimalBin()

    x = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
                    stats.cauchy(-4, 0.8).rvs(2000),
                    stats.cauchy(-1, 0.3).rvs(500),
                    stats.cauchy(2, 0.8).rvs(1000),
                    stats.cauchy(4, 1.5).rvs(500)])

    # truncate values to a reasonable range
    x = x[(x > -15) & (x < 15)]

    y = np.ones_like(x)

    plt.hist(x, density=True, bins=200, histtype="stepfilled", alpha=0.2)
    binner.fit(x, y)

    plt.hist(x, density=True, bins=binner.bins, histtype="step")

    plt.show()
