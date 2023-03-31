import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import glmnet_python
from glmnet_python import glmnet
import sys
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot


# データ生成
N = 100
p = 2
X = np.random.randn(N, p)
X = np.concatenate([np.ones(N).reshape(N, 1), X], axis=1)
beta = np.random.randn(p+1)
y = np.zeros(N)
s = np.dot(X, beta)
prob = 1 / (1 + np.exp(s))
for i in range(N):
    if 1/2 > prob[i]:
        y[i] = 1
    else:
        y[i] = -1
beta


# 最尤推定値の計算
beta = np.inf
gamma = np.random.randn(p + 1)
while np.sum((beta - gamma) ** 2) > 0.001:
    beta = gamma.copy()
    s = np.dot(X, beta)
    v = np.exp(-s * y)
    u = y * v / (1 + v)
    w = v / (1 + v) ** 2
    z = s + u / w
    W = np.diag(w)
    gamma = np.dot(np.linalg.inv(X.T @ W @ X), np.dot(X.T @ W, z))       ##
    print(gamma)
beta  # 真の値。最尤法でこの値を推定したい

# Linuxマシンのフォルダに"breastcancer.csv"をおく
x = np.loadtxt("breastcancer.csv",
               delimiter=",", skiprows=1, usecols=range(1000))
y = np.loadtxt("breastcancer.csv",
               delimiter=",", skiprows=1, dtype="unicode", usecols=1000)
n = len(y)
yy = np.ones(n)
for i in range(n):
    if y[i] == "control":
        yy[i] = 1
    else:
        yy[i] = -1
fit1 = cvglmnet(x=x.copy(), y=yy.copy(), ptype="deviance", family="binomial")
fit2 = cvglmnet(x=x.copy(), y=yy.copy(), ptype="class", family="binomial")
beta = cvglmnetCoef(fit1)
np.sum(beta != 0)

# CVのグラフを作成する
fig = plt.figure()
cvglmnetPlot(fit1)
fig.savefig("img1.png")
fig2 = plt.figure()
cvglmnetPlot(fit2)
fig2.savefig("img2.png")
# Linuxマシンのフォルダに"img1.png", "img2.png"ができている