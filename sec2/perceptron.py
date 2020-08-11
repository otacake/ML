import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

class Perceptron(object):
    def __init__(self,eta,n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self,X,y):
        self.errors_ = []
        self.w_ = np.zeros(1+X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                yi = self.predict(xi)
                dw = self.eta * (target - yi)
                self.w_[1:] += dw*xi
                self.w_[0] += dw
                if dw != 0.0:
                    errors +=1
            self.errors_.append(errors)
        return self



    def net_input(self,x):
        return np.dot(x,self.w_[1:]) + self.w_[0]

    def predict(self,x):
        return np.where(self.net_input(x)>=0.0,1,-1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)
X = df.iloc[0:100,[0,2]].values

ppn = Perceptron(eta=0.01,n_iter=10)
ppn.fit(X,y)
plot_decision_regions(X,y,ppn)

plt.show()
