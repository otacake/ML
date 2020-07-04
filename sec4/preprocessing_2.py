import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.DataFrame([["green","M",10.1,"class1"],["red","L",13.5,"class2"],["blue","XL",15.3,"class1"]])
df.columns = ["colour","size","price","classlabel"]

#順序特徴量のマッピング
size_mapping = {"XL":3,"L":2,"M":1}
df["size"] = df["size"].map(size_mapping)

#クラスラベルのマッピング
class_mappimg = {label:idx for idx,label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mappimg)
print(df)
#名義特徴量のエンコーディング

df = pd.get_dummies(df,drop_first=True)
print(df)
#特徴量の尺度を揃える
from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler()
std = StandardScaler()
X = df.iloc[:,1:2].values
X_nor = mms.fit_transform(X)
print(X_nor)
print(std.fit_transform(X))