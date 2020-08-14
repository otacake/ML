import numpy as np
import pandas as pd
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1,random_state=0)
lr.fit(X_train_std,y_train)
y_pred = lr.predict(X_test_std)

n = len(y_test)
correct = 0
for i in range(n):
    if y_pred[i] == y_test[i]:
        correct +=1
print(correct*100/n)