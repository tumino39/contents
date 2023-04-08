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
from sklearn.datasets import load_iris

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


def gr_multi_lasso(X, y, lam):
    n = X.shape[0]
    p = X.shape[1]
    K = len(np.unique(y))
    beta = np.ones((p, K))
    Y = np.zeros((n, K))
    for i in range(n):
        Y[i, y[i]] = 1
    eps = 1
    while eps > 0.001:
        gamma = copy.copy(beta)
        eta = X @ beta
        P = np.exp(eta)
        for i in range(n):
            P[i, ] = P[i, ] / np.sum(P[i, ])
        t = 2 * np.max(P*(1-P))
        R = (Y-P) / t
        for j in range(p):
            r = R + X[:, j].reshape(n, 1) @ beta[j, :].reshape(1, K)
            M = X[:, j] @ r
            beta[j, :] = (max(1 - lam / t / np.sqrt(np.sum(M*M)), 0)
                          / np.sum(X[:, j]*X[:, j]) * M)
            R = r - X[:, j].reshape(n, 1) @ beta[j, :].reshape(1, K)
        eps = np.linalg.norm(beta - gamma)
    return beta

iris = load_iris()
X = np.array(iris["data"])
y = np.array(iris["target"])

lambda_seq = np.arange(10, 151, 10)
m = len(lambda_seq)
p = X.shape[1]
K = 3
alpha = np.zeros((m, p, K))
for i in range(m):
    res = gr_multi_lasso(X, y, lambda_seq[i])
    alpha[i, :, :] = res
plt.xlim(0, 150)
plt.ylim(np.min(alpha), np.max(alpha))
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
handles = []
labels = ["がく片の長さ", "がく片の幅", "花びらの長さ", "花びらの幅"]
cols = ["red", "green", "blue", "cyan"]
for i in range(4):
    for k in range(K):
        line, = plt.plot(lambda_seq, alpha[:, i, k], color=cols[i],
                         label="{}".format(labels[i]))
    handles.append(line)
plt.legend(handles, labels, loc="upper right")

plt.show()
