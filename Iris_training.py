from perceptron import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)
X = df.iloc[0:100,[0,2]].values

ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

ya = df.iloc[101:,4].values
xa = df.iloc[101:,[0,2]].values

for xi,real in zip(xa,ya):
    ans = ppn.predict(xi)
    pre = "Iris-setosa"
    if ans >= 0.0:
        pre = "Iris-verginica"
    print("predict is " +pre, "the ans is",real)

