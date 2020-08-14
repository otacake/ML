import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.DataFrame([["green","M",10.1,"class1"],["red","L",13.5,"class2"],["blue","XL",15.3,"class1"]])
df.columns = ["colour","size","price","classlabel"]

size_mapping = {"M":1,"L":2,"XL":3}
df["size"] = df["size"].map(size_mapping)

class_mapping = {label:idx for idx,label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)

df = pd.get_dummies(df)
print(df)

