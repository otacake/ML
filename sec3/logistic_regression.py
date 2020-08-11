import numpy as np

class LogisticRegressionGD(object):

    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter = n_iter
        self.random_state = random_state

    def activation(self,z): #シグモイド関数
        return 1./(1. + np.exp(-np.clip(z,-250,250))) #np.clipでzの範囲を指定

    def net_input(self,X): #内積というよりもむしろ行列の積。Xごと掛けるから
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,0)

    def fit(self,X,y): #学習
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output) #yベクトルとの差のベクトル

            self.w_[1:] += self.eta * X.T.dot(errors) #差分の所,転置がキモ
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None,resolution=0.02):
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

    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker="o",s=55,label="test set")


iris = datasets.load_iris()

#iris.dataで特徴量を出せる
#iris.targetでクラスラベルのベクトルを出せる、もう数値化されてる

X = iris.data[:,[2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#ここから特徴量の標準化をする
from sklearn.preprocessing import StandardScaler #オブジェクト
sc = StandardScaler()
sc.fit(X_train) #ここで平均と標準偏差をベクトルごとに出してる
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

from sklearn.svm import SVC
svm = SVC(kernel="rbf",C=1.0,gamma=0.2,random_state=0)

svm.fit(X_train_std,y_train)

plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.show()