import copy
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

def centralize(X0, y0, standardize=True):
    X = copy.copy(X0)
    y = copy.copy(y0)
    n, p = X.shape
    X_bar = np.zeros(p)                   # Xの各列の平均
    X_sd = np.zeros(p)                    # Xの各列の標準偏差
    for j in range(p):
        X_bar[j] = np.mean(X[:, j])
        X[:, j] = X[:, j] - X_bar[j]      # Xの各列の中心化
        X_sd[j] = np.std(X[:, j])
        if standardize is True:
            X[:, j] = X[:, j] / X_sd[j]   # Xの各列の標準化
    if np.ndim(y) == 2:
        K = y.shape[1]
        y_bar = np.zeros(K)               # yの平均
        for k in range(K):
            y_bar[k] = np.mean(y[:, k])
            y[:, k] = y[:, k] - y_bar[k]  # yの中心化
    else:                                 # yがベクトルの場合
        y_bar = np.mean(y)
        y = y - y_bar
    return X, y, X_bar, X_sd, y_bar

def gr_multi_linear_lasso(X, Y, lam):
    n, p = X.shape
    K = Y.shape[1]
    X, Y, x_bar, x_sd, y_bar = centralize(X, Y)
    beta = np.zeros((p, K))
    gamma = np.zeros((p, K))
    eps = 1
    while eps > 0.01:
        gamma = copy.copy(beta)
        R = Y - X @ beta
        for j in range(p):
            r = R + X[:, j].reshape(n, 1) @ beta[j, :].reshape(1, K)
            M = X[:, j] @ r
            beta[j, :] = (max(1 - lam / np.sqrt(np.sum(M*M)), 0)
                          / np.sum(X[:, j] * X[:, j]) * M)
            R = r - X[:, j].reshape(n, 1) @ beta[j, :].reshape(1, K)
        eps = np.linalg.norm(beta - gamma)
    for j in range(p):
        beta[j, :] = beta[j, :] / x_sd[j]
    beta_0 = y_bar - x_bar @ beta
    return [beta_0, beta]

df2 = np.loadtxt('central_2.txt', delimiter=",",encoding="utf_8_sig")
index2 = list(set(range(18)) - {5,6})
X2 = np.array(df2[:, index2])
Y2 = np.array(df2[:, [5, 6]])
lambda_seq = np.arange(0, 200, 5)
m = len(lambda_seq)
beta_12 = np.zeros((m, 16))
beta_22 = np.zeros((m, 16))

for i in range(m):
    beta2 = gr_multi_linear_lasso(X2, Y2, lambda_seq[i])
    beta_12[i, :] = beta2[1][:, 0]
    beta_22[i, :] = beta2[1][:, 1]
beta_max2 = np.max(np.array([beta_12, beta_22]))
beta_min2 = np.min(np.array([beta_12, beta_22]))

plt.xlim(0, 200)
plt.ylim(beta_min2, beta_max2)
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels2 = ["打率","試合","打席数","打数","安打", "盗塁", "四球", "死球", "三振", "犠打", "併殺打","出塁率","長打率","OPS","RC27","XR27"]
lins = ["solid", "dashed"]
cols2 = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray","black"]
for i in range(8):
    plt.plot(lambda_seq, beta_12[:, i], color=cols2[i], linestyle=lins[0],
             label="{}".format(labels2[i]))
    plt.plot(lambda_seq, beta_22[:, i], color=cols2[i], linestyle=lins[1],
             label="{}".format(labels2[i]))
plt.legend(loc="upper right")
plt.show()
for i in range(8,15):
    t = i-8
    plt.plot(lambda_seq, beta_12[:, i], color=cols2[t], linestyle=lins[0],
             label="{}".format(labels2[i]))
    plt.plot(lambda_seq, beta_22[:, i], color=cols2[t], linestyle=lins[1],
             label="{}".format(labels2[i]))
plt.legend(loc="upper right")
plt.show()
