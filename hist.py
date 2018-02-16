import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gammaln

def lnL_hogg(n_bins, x):

        a = 10

        N, e = np.histogram(x, bins=n_bins)
        d = e[1] - e[0]

        s = np.sum(N + a)
        L = np.sum(N * np.log((N + a - 1) / (d * (s - 1))))

        return L


def lnL(m, x):

        N = len(x)

        n = np.histogram(x, bins=m)[0]

        p1 = N * np.log(m) + gammaln(m/2) - gammaln(N + m/2)
        p2 = -m * gammaln(0.5) + np.sum(gammaln(n + 0.5))

        return p1 + p2


def optimal_binning(x):

        bins = np.linspace(2, 100).astype(int)
        ls = np.zeros_like(bins)
        for i, b in enumerate(bins):
                ls[i] = lnL_hogg(b, x)

        return bins[ls.argmax()], ls


if __name__ == '__main__':

        df = pd.read_csv('./weight.txt')
        b, l = optimal_binning(df.weight.values)
        df.weight.hist(bins=b)
        plt.show()
