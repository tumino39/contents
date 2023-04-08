import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import time
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import warnings
warnings.simplefilter('ignore', FutureWarning)



def R2(x, y):
    model = LinearRegression()
    model.fit(x, y)           # モデルの訓練
    y_hat = model.predict(x)  # 予測値の表示
    y_bar = np.mean(y)
    RSS = np.dot(y - y_hat, y - y_hat)
    TSS = np.dot(y - y_bar, y - y_bar)
    return 1 - RSS / TSS

def vif(x):
    p = x.shape[1]
    values = np.zeros(p)
    for j in range(p):
        ind = [i for i in range(p) if i != j]
        values[j] = 1 / (1 - R2(x[:, ind], x[:, j]))
    return values

boston = load_boston()
n = boston.data.shape[0]
z = np.concatenate([boston.data, boston.target.reshape([n, 1])], 1)
print(vif(z))


