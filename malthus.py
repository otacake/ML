
import numpy as np # NumPyモジュール
import matplotlib.pyplot as plt  # グラフ表示モジュール

# 表示する関数
def func_f(t, c, t0, num0):
    return num0*np.exp(c*(t-t0))

# グラフ化の関数（副プログラム）
def plotfunc(func_f, c, t0, num0, t_min, t_max, coulor):
    plt.title("1800年〜1900年のデータから予測した世界人口の変化", fontname="Hiragino Maru Gothic Pro")
    # Windowsで日本語を表示するには，fontname="MS Gothic" とする．
    # x軸の名前．$で囲むと斜体になる
    plt.xlabel("西暦［年］", fontname="Hiragino Maru Gothic Pro")
    # y軸の名前
    plt.ylabel("人口［億人］", fontname="Hiragino Maru Gothic Pro")
    plt.xlim(t_min, t_max)    # x軸の表示範囲
    plt.grid(True)    # グリッドを表示
    plt.minorticks_on()    # 細かいグリッドを追加表示
    plt.grid(True, which='minor', alpha=0.2)

    # 表示年数t=t_min〜t_maxを500等分した値をもつ配列構造(ndarray)を作成
    delta_t = (t_max - t_min)/500.0
    t = np.arange(t_min, t_max+delta_t, delta_t)
    num = func_f(t, c, t0, num0)

    # 関数を折線グラフで表示
    plt.plot(t, num, label="c={0}".format(c), color=coulor)
    # グラフの色はカラーコード'#ff0000'の指定により赤となる．
    # （例）緑：#00ff00，青：#0000ff，黄：#ffff00，黒：#000000  # グラフを表示

# 主プログラム
if (__name__ == '__main__'):
    c_array = [0.00521,0.0,-0.00521]    # マルサス係数
    color_array = ["#ff0000","#00ff00","#0000ff"]
    t0 = 1800        # 基準となる年（西暦）
    num0 = 9.8     # t0における世界人口
    t_min = 1800
    t_max = 1900
    for i in range(3):
        plotfunc(func_f, c_array[i], t0, num0, t_min, t_max,color_array[i])
    plt.legend()
    plt.show()

# END
