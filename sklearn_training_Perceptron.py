from sklearn import datasets
import numpy as np

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

#ここまででようやく下準備ができたといえるみたい？(実際はもっと大変)

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40,eta0=0.01,random_state=1)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)

count = len(y_test)
miss = 0
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        miss+=1
print("missclasified samples:",str(miss))
print("MIsclassified rate:",str(miss/count))