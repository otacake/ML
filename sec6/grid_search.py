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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

param_range = [float(10**i) for i in range(-4,4)]

pipe_svm = Pipeline([("std",StandardScaler()),("clf",SVC(random_state=1))])

param_grid = [{"clf__C":param_range,"clf__kernel":['linear']},{"clf__C":param_range,"clf__gamma":param_range,"clf__kernel":['rbf']}]
gs = GridSearchCV(estimator=pipe_svm,param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

gs = gs.fit(X_train,y_train)

clf = gs.best_estimator_
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))