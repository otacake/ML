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
count = 0
loop = 0
for xi,real in zip(xa,ya):
    loop +=1
    ans = ppn.predict(xi)
    pre = "Iris-setosa"
    if ans >= 0.0:
        pre = "Iris-virginica"
    print("predict is " +pre, "the ans is",real)
    if pre == real:
        count +=1

print("--------------------")
print("精度は"+str((count/loop)*100)+"%です!")
print("--------------------")
