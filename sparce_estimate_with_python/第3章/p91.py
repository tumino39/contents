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

df = np.loadtxt('giants_2019.txt', delimiter="\t")
index = list(set(range(9)) - {1, 2})
X = np.array(df[:, index])
Y = np.array(df[:, [1, 2]])
lambda_seq = np.arange(0, 200, 5)
m = len(lambda_seq)
beta_1 = np.zeros((m, 7))
beta_2 = np.zeros((m, 7))

for i in range(m):
    beta = gr_multi_linear_lasso(X, Y, lambda_seq[i])
    beta_1[i, :] = beta[1][:, 0]
    beta_2[i, :] = beta[1][:, 1]
beta_max = np.max(np.array([beta_1, beta_2]))
beta_min = np.min(np.array([beta_1, beta_2]))

plt.xlim(0, 200)
plt.ylim(beta_min, beta_max)
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["安打", "盗塁", "四球", "死球", "三振", "犠打", "併殺打"]
cols = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray"]
lins = ["solid", "dashed"]
for i in range(7):
    plt.plot(lambda_seq, beta_1[:, i], color=cols[i], linestyle=lins[0],
             label="{}".format(labels[i]))
    plt.plot(lambda_seq, beta_2[:, i], color=cols[i], linestyle=lins[1],
             label="{}".format(labels[i]))
plt.legend(loc="upper right")
plt.show()


df2 = np.loadtxt('central.txt', delimiter=",",encoding="utf_8_sig")
index = list(set(range(9)) - {1, 2})
X = np.array(df2[:, index])
Y = np.array(df2[:, [1, 2]])
lambda_seq = np.arange(0, 500, 5)
m = len(lambda_seq)
beta_12 = np.zeros((m, 7))
beta_22 = np.zeros((m, 7))

for i in range(m):
    beta2 = gr_multi_linear_lasso(X, Y, lambda_seq[i])
    beta_12[i, :] = beta2[1][:, 0]
    beta_22[i, :] = beta2[1][:, 1]
beta_max2 = np.max(np.array([beta_12, beta_22]))
beta_min2 = np.min(np.array([beta_12, beta_22]))

plt.xlim(0, 400)
plt.ylim(beta_min2, beta_max2)
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["安打", "盗塁", "四球", "死球", "三振", "犠打", "併殺打"]
cols = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray"]
lins = ["solid", "dashed"]
for i in range(7):
    plt.plot(lambda_seq, beta_12[:, i], color=cols[i], linestyle=lins[0],
             label="{}".format(labels[i]))
    plt.plot(lambda_seq, beta_22[:, i], color=cols[i], linestyle=lins[1],
             label="{}".format(labels[i]))
plt.legend(loc="upper right")
plt.show()