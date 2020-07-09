import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
 'machine-learning-databases/wine/wine.data',
 header=None)
#wineのデータ

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y,
                     test_size=0.3,
                     random_state=0,
                     stratify=y)

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)
importance=forest.feature_importances_
dic = {}
for i in range(len(importance)):
    dic[importance[i]] = feat_labels[i]

importance.sort()
na = len(importance)
for i in range(len(importance)):
    print(dic[importance[na-1-i]] +":"+ str(importance[na-1-i]))

#特徴量の重要度を出してくれる