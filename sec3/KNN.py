import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

#iris.dataで特徴量を出せる
#iris.targetでクラスラベルのベクトルを出せる、もう数値化されてる

X = iris.data[:,[2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
#トレーニングデータとテストデータを分けている
#この書き方だと3割がテストに
#random_stateによって内部で生成される乱数の調整
#この乱数の調整によってデータの分割に再現性が生まれる
#stratifyによってトレーニングデータにも、テストデータにも出てくるラベルの比率が指定したベクトルの比率と同じになる

#ここから特徴量の標準化をする
from sklearn.preprocessing import StandardScaler #オブジェクト
sc = StandardScaler()
sc.fit(X_train) #ここで平均と標準偏差をベクトルごとに出してる
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#KNN
#計算量がバカでかくなることも
#KNNは多数決
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric="minkowski")
knn.fit(X_train_std,y_train)
X_combined = np.vstack((X_train,X_test)) #下にくっつける、行列の形を考えれば自然
y_combined = np.hstack((y_train,y_test))

from library_ML import plot_decision_regions
plot_decision_regions(X_train_std,y_train,classifier=knn)
plt.show()