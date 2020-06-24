import numpy as np
import pandas as pd

class ADALINE(object):

    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter = n_iter
        self.random_state = random_state

    def activation(self,X): #線形活性化関数
        return X #これはADALINEのとき

    def net_input(self,X): #ADALINEでは内積というよりもむしろ行列の積。Xごと掛けるから
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)

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
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
