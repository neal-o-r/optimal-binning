import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.special import gammaln


class OptimalBin(BaseEstimator, TransformerMixin):
        '''
        Take in a 1d array and a set of binary labels (x, y)
        and return the mean y in each of a set of
        'optimal' bins
        '''

        def __init__(self, a='auto', max_bins=100):
                self.max_bins = max_bins
                self.a = a


        def _lnL(self, n_bins, x, y):
                '''
                Log likelihood from Hogg 2008
                '''
                # this is totally arbitrary, should fit
                a = np.mean(y) if self.a is 'auto' else self.a

                N, e, _ = stats.binned_statistic(x, statistic='sum',
                                values=y, bins=n_bins)
                d = e[1] - e[0]

                if any((N + a) < 1):
                        return np.nan

                s = np.sum(N + a)
                L = np.sum(N * np.log((N + a - 1) / (d * (s - 1))))

                return L


        def _optimal_bin_no(self, x, y):
                bins = np.linspace(2, self.max_bins).astype(int)
                ls = []
                for b in bins:
                        i = self._lnL(b, x, y)
                        if np.isnan(i):
                                break
                        ls.append(i)

                return bins[np.argmax(ls)]


        def fit(self, x, y):
                self.bin_no = self._optimal_bin_no(x, y)
                self.bins = np.histogram(x, bins=self.bin_no)[1]

                avg = lambda a: np.sum(a) / len(a) if len(a) else 0
                self.mu = stats.binned_statistic(x, statistic=avg,
                                values=y, bins=self.bins)[0]

                return self


        def transform(self, x):
                locs = np.digitize(x, self.bins)
                # np hist and digitize handle bins differently
                # so this fudge is required (numpy/issues/9208)
                locs -= 1
                locs[locs == len(self.mu)] -= 1
                return self.mu[locs]


        def fit_transform(self, x, y):
                self.fit(x, y)
                return self.transform(x)

