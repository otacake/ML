import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
X = df.iloc[:,2:].values
y = df.iloc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)
from sklearn.tree import DecisionTreeClassifier as Tree

tree = Tree(criterion="entropy",max_depth=2,random_state=0)

tree.fit(X_train,y_train)

print(tree.score(X_test,y_test))

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=0)
ada.fit(X_train,y_train)
print(ada.score(X_test,y_test))