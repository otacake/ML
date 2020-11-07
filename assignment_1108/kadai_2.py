# CSVファイル（カンマ区切り）を読み込む

import numpy as np	# NumPyモジュール
import matplotlib.pyplot as plt  # グラフ表示モジュール

# グラフの共通した装飾（副プログラム）
def plotinit(st_max, mol_min, mol_max):
	plt.title("SDS水溶液の表面張力", fontname="Hiragino Maru Gothic Pro")
	# Windowsで日本語を表示するには，fontname="MS Gothic" とする．

	# x軸の名前．$で囲むと斜体になる
	plt.xlabel("SDS濃度［mol/L］", fontname="Hiragino Maru Gothic Pro")
	# y軸の名前
	plt.ylabel("表面張力［dyn/cm］", fontname="Hiragino Maru Gothic Pro")
	plt.xlim(mol_min, mol_max)			# x軸の表示範囲
	plt.ylim(0,st_max)					# y軸の表示範囲
	plt.grid(True)							# グリッドを表示
	plt.minorticks_on()					# 細かいグリッドを追加表示
	plt.grid(True, which='minor', alpha=0.2)

def approximate_function(x,y):
	coef = np.polyfit(x,y,3)
	return coef

# 主プログラム
if (__name__ == '__main__'):

	f_path = './st_stat.data'

	st_stat = np.loadtxt(f_path)		# ファイルを読み込み
	mol = st_stat[0, :]					# 1行目のデータ(SDS濃度)を抽出
	st_mean = st_stat[1, :]				# 2行目のデータ(表面張力の平均値)を抽出
	st_std = st_stat[2, :]				# 3行目のデータ(表面張力の標準偏差)を抽出

# グラフの描画
# グラフの共通した装飾を実行
	st_max = 100.0
	mol_min = 0.0
	mol_max = 0.01
	plotinit(st_max, mol_min, mol_max)

# 散布図の描画
	#plt.scatter(mol, st_mean, color="#ff0000", s = 40)
	plt.errorbar(mol,st_mean,yerr=st_std,capsize=5,fmt='o',markersize=8,ecolor='black', markeredgecolor = "black", color='w')
	coef = approximate_function(mol,st_mean)
	x = np.linspace(0,0.010,1000000)
	y = coef[0]*x**3 + coef[1]*x**2 + coef[2]*x + coef[3]
	plt.plot(x,y)
	plt.show()

# END
