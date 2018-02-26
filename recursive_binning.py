import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


class RecursiveOptimalBin(BaseEstimator, TransformerMixin):
        '''
        Take in a 1d array and a set of binary labels (x, y)
        and return the mean y in each of a set of
        'optimal' bins
        '''
        def __init__(self, a=10, max_bins=100, method='pdf'):
                '''
                Init with 3 args:
                        a        : smoothing factor
                        max_bins : most bins that will be tested
                        method   : pdf will return a normalised pdf
                                        - sum y in bin/total sum y
                                   avg will return
                                        - mean of y bin
                '''
                assert (method is 'pdf' or
                        method is 'avg')

                self.method = method
                self.max_bins = max_bins
                self.a = a


        def _lnL(self, bins, x, y):
                '''
                Log likelihood from Hogg 2008
                '''

                N, e, _ = stats.binned_statistic(x, statistic='sum',
                                values=y, bins=bins)

                d = (bins - np.roll(bins, 1))[1:]

                if any((N + self.a) < 1):
                        return np.nan

                s = np.sum(N + self.a)
                L = np.sum(N * np.log((N + self.a - 1) / (d * (s - 1))))

                return L


        def optimal_binning(self, x, y):
                '''
                Find the optimal binning for a data set
                Begin with 100 bins, and keep combining
                adjacent bins, so long as it increases the
                log-likelihood.
                '''
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
                        for i in range(1, len(bins)-1):
                                test_bins = bins[:i] + bins[i+1:]
                                Ls.append(self._lnL(test_bins, x, y))

                        L_new = np.nanmax(Ls)
                        L_ind = 1 + np.nanargmax(Ls)
                        bins_new = bins[:L_ind] + bins[L_ind+1:]

                return bins


        def fit(self, x, y):
                '''
                Get the optimal bin number and create those bins,
                then aggregate the labels accordingly and populate
                self.mu
                '''
                self.bins = self.optimal_binning(x, y)

                avg = lambda a: np.sum(a) / len(a) if len(a) else 0
                pdf = lambda a: np.sum(a) / np.sum(y)

                if self.method is 'avg':
                        agg = avg
                else:
                        agg = pdf

                self.mu = stats.binned_statistic(x, statistic=agg,
                                values=y, bins=self.bins)[0]
                return self


        def transform(self, x):
                '''
                get the bin that each x fell into and return
                the mu in corresponding to that bin
                '''
                locs = np.digitize(x, self.bins)
                # np hist and digitize handle bins differently
                # so this fudge is required (numpy/issues/9208)
                locs -= 1
                locs[locs == len(self.mu)] -= 1
                return self.mu[locs]


        def fit_transform(self, x, y):
                self.fit(x, y)
                return self.transform(x)


