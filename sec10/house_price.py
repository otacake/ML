import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


cols = ["LSTAT","INDUS","RM","MEDV"]

X = df[["RM"]].values
y = df["MEDV"].values

def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color="red")
    return None

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
lin_regplot(X,y,slr)
plt.show()