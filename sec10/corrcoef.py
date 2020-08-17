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

sns.set(style='whitegrid',context='notebook')
cols = ["LSTAT","INDUS","RM","MEDV"]

cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,xticklabels=cols,yticklabels=cols)
plt.show()