import numpy as np
import matplotlib.pyplot as plt
from library_ML import plot_decision_regions
from sklearn import datasets

iris = datasets.load_iris()

#iris.dataで特徴量を出せる
#iris.targetでクラスラベルのベクトルを出せる、もう数値化されてる

X = iris.data[:,[2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

from sklearn.tree import DecisionTreeClassifier #決定木のオブジェクト生成
tree = DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=1) #ジニ不純度を指標に、深さの最大を4までに設定、state指定して再現性を持たせる
tree.fit(X_train,y_train) #学習作業
X_combined = np.vstack((X_train,X_test)) #下にくっつける、行列の形を考えれば自然
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.show()