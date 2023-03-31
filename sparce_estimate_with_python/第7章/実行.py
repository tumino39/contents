import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import pandas as pd
import time
import copy as c
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA


def gr(X, y, lam):
    p = X.shape[1]
    nu = 1 / np.max(np.linalg.eigvals(X.T @ X))
    beta = np.zeros(p)
    beta_old = np.zeros(p)
    eps = 1
    while eps > 0.001:
        gamma = beta + nu * X.T @ (y - X @ beta)
        beta = max(1 - lam * nu / np.linalg.norm(gamma, 2), 0) * gamma
        eps = np.max(np.abs(beta - beta_old))
        beta_old = copy.copy(beta)
    return beta

def soft_th(lambd, x):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

# データ生成
n = 100
p = 15
x = np.random.normal(size=n*p).reshape(-1, p)
np.savetxt("研究室\第7章\X.csv", x, delimiter=",")
# x = pd.read_csv('研究室\第7章\mnist1.csv',sep = "," , header= None).to_numpy()

lambd_seq = np.arange(0, 11) / 10
lambd = [0.00001, 0.001]
m = 100
g = np.zeros((m, p))
Z1 =[0] * 5
Z2 =[0] * 5

# グラフ表示
g_max = np.max(g)
g_min = np.min(g)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 日本語を表示するため
plt.rcParams["axes.unicode_minus"] = False    # -を表示するため
fig = plt.figure(figsize=(10, 5))


pca = PCA(n_components=1)
pca.fit(x)



for kai in range(5):
    # lambda = 0.00001
    # u,vの計算
    for j in range(p):
        x[:, j] = x[:, j] - np.mean(x[:, j])
    for j in range(p):
        x[:, j] = x[:, j] / np.sqrt(np.sum(np.square(x[:, j])))
    r = [0] * n
    v = np.random.normal(size=p)
    for h in range(m):
        z = np.dot(x, v)
        u = np.dot(x.T, z)
        if np.sum(np.square(u)) > 0.00001:
            u = u / np.sqrt(np.sum(np.square(u)))
        for k in range(p):
            m1 = list(np.arange(k))
            n1 = list(np.arange(k+1, p))
            z = m1 + n1
            for i in range(n):
                r[i] = (np.sum(u * x[i, :])
                        - np.sum(np.square(u)) * sum(x[i, z] * v[z]))
            S = np.sum(np.dot(x[:, k], r)) / n
            v[k] = soft_th(lambd[0], S)
        if np.sum(np.square(v)) > 0.00001:
            v = v / np.sqrt(np.sum(np.square(v)))
        g[h, :] = v
    if g[99,4]< 0:
       for i in range(100):
        g[i,:] = -g[i,:]
    Z1[kai] = g
    print(g[99,:])
    # 作図
    plt.subplot(1, 2, 1)
    plt.title(r"$\lambda = 0.00001$")
    plt.xlabel("繰り返し回数")
    plt.ylabel("$v$の各要素")
    for j in range(p):
        plt.plot(range(1, m+1), g[:, j])
    

    # lambda = 0.001
    # u,vの計算
    for j in range(p):
        x[:, j] = x[:, j] - np.mean(x[:, j])
    for j in range(p):
        x[:, j] = x[:, j] / np.sqrt(np.sum(np.square(x[:, j])))
    r = [0] * n
    v = np.random.normal(size=p)
    for h in range(m):
        z = np.dot(x, v)
        u = np.dot(x.T, z)
        if np.sum(np.square(u)) > 0.00001:
            u = u / np.sqrt(np.sum(np.square(u)))
        for k in range(p):
            m1 = list(np.arange(k))
            n1 = list(np.arange(k+1, p))
            z = m1 + n1
            for i in range(n):
                r[i] = np.sum(u * x[i, :]) - np.sum(np.square(u)) * sum(x[i, z] * v[z])
            S = np.sum(np.dot(x[:, k], r)) / n
            v[k] = soft_th(lambd[1], S)
        if np.sum(np.square(v)) > 0.00001:
            v = v / np.sqrt(np.sum(np.square(v)))
        g[h, :] = v
    if g[99,4]< 0:
       for i in range(100):
        g[i,:] = -g[i,:]
    Z2[kai] = g
    print(g[99,:])
    # 作図
    plt.subplot(1, 2, 2)
    plt.title(r"$\lambda = 0.001$")
    plt.xlabel("繰り返し回数")
    plt.ylabel("$v$の各要素")
    for j in range(p):
        plt.plot(range(1, m+1), g[:, j])
    plt.tight_layout()  # 図の重なりを回避
    plt.show()

