import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from binning import OptimalBin

if __name__ == '__main__':

        df = pd.read_csv('iris.csv')
        df['label'] = (df.species == 'setosa').astype(int)
        binner = OptimalBin()

        x = df.sepal_length.values
        y = df.label.values

        mu = binner.fit_transform(x, y)
