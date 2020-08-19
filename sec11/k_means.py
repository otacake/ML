import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=150,centers=3,cluster_std=0.5,random_state=0)
"""
plt.scatter(X[:,0],X[:,1],c="black",marker="o",s=50)
plt.grid()
plt.show()
"""
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=0,init="random")
y_km = km.fit_predict(X)
li = [0.0,1.0,2.0]
dic1 = {0.0:'lightgreen',1.0:'orange',2.0:'lightblue'}
dic2 = {0.0:'s',1.0:'o',2.0:'v'}
for i in range(3):
    lab = "cluster " + str(i+1)
    plt.scatter(X[y_km==li[i],0],X[y_km==li[i],1],s=50,c=dic1[li[i]],marker=dic2[li[i]],label=lab)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red')
plt.legend()
plt.grid()
plt.show()