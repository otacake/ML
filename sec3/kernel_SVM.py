#XORの前の下準備(XORは大丈夫やけど、TT->0,TF->1,FT->1,FF->0のやつね)
import numpy as np
import matplotlib.pyplot as plt
from library_ML import plot_decision_regions

np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)

#ランダムに生成したXORについてグラフを書いた
#生成されるグラフを見ると明らかだけど、線形分離は不可
#じゃあどうする?
#次元上げちゃえばいい！
#次元を上げると計算が大変なことになるんです(手計算したひとならわかる)
#だったら一番だるい内積計算を楽しませんか、カーネル関数で

from sklearn.svm import SVC
svm = SVC(kernel="rbf",random_state=1,gamma=0.05,C=10.0)
svm.fit(X_xor,y_xor) #学習
plot_decision_regions(X_xor,y_xor,classifier=svm) #グラフへの描写
plt.show()

#gammaは大きいほどfitが強くなる