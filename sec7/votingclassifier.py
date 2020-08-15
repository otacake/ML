from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = datasets.load_iris()
X,y = df.data[50:,[1,2]],df.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipe1 = Pipeline([("std",StandardScaler()),("clf",LogisticRegression(random_state=0,C=0.001,penalty='l2'))])
tree = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)
pipe2 = Pipeline([("std",StandardScaler()),("clf",KNN(n_neighbors=3))])

est = [("lr",pipe1),("tree",tree),("KNN",pipe2)]

vc_clf = VotingClassifier(estimators=est,n_jobs=-1)

scores1 = cross_val_score(estimator=vc_clf,X=X_train,y=y_train,cv=10,n_jobs=-1)
scores2 = cross_val_score(estimator=pipe2,X=X_train,y=y_train,cv=10)
print(np.mean(scores1))
print(np.mean(scores2))
