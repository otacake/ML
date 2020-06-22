import numpy as np

class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta # 学習率
        self.n_iter = n_iter # 試行回数
        self.random_state = random_state # 重み初期化乱数

    def net_input(self,X): #総入力(内積)
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0,1,-1)

    def fit(self,X,y): #学習
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size = 1+X.shape[1]) #いっちゃん最初の係数を乱数で定めてる
        self.errors_ = []

        for _ in range(self.n_iter): #学習
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

