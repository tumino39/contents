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

def group_lasso(z, y, lam=0):
    J = len(z)
    theta = []
    for i in range(J):
        theta.append(np.zeros(z[i].shape[1]))
    for m in range(10):
        for j in range(J):
            r = copy.copy(y)
            for k in range(J):
                if k != j:
                    r = r - z[k] @ theta[k]
            theta[j] = gr(z[j], r, lam)
    return theta

n = 100
J = 2
u = randn(n)
v = u + randn(n)
s = 0.1 * randn(n)
t = 0.1 * s + randn(n)
b3 = 0.5*v + 0.5*t + randn(n)
y = u + v + s + t + b3 + randn(n)
z = []
z = np.array([np.array([u, v,b3]).T, np.array([b3,s, t]).T])
lambda_seq = np.arange(0, 500, 10)
m = len(lambda_seq)
beta = np.zeros((m, 6))
for i in range(m):
    est = group_lasso(z, y, lambda_seq[i])
    beta[i, :] = np.array([est[0][0], est[0][1],est[0][2],est[1][1], est[1][2],est[1][0]])
plt.xlim(0, 500)
plt.ylim(np.min(beta), np.max(beta))
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["グループ1", "グループ1", "グループb3","グループ2", "グループ2",]
cols = ["red", "blue","yellow"]
lins = ["solid", "dashed"]
plt.plot(lambda_seq, beta[:, 0], color=cols[0],
             linestyle=lins[0], label="{}".format(labels[0]))
plt.plot(lambda_seq, beta[:, 1], color=cols[0],
             linestyle=lins[1], label="{}".format(labels[1]))
plt.plot(lambda_seq, beta[:, 3], color=cols[1],
             linestyle=lins[0], label="{}".format(labels[3]))
plt.plot(lambda_seq, beta[:, 4], color=cols[1],
             linestyle=lins[1], label="{}".format(labels[4]))
plt.plot(lambda_seq, beta[:, 2], color=cols[2],
             linestyle=lins[0], label="{}".format(labels[2]))
plt.plot(lambda_seq, beta[:, 5], color=cols[2],
             linestyle=lins[1], label="{}".format(labels[2]))
plt.legend(loc="upper right")
plt.axvline(0, color="black")
plt.show()