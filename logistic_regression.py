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
