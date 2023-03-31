import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

def fista(X, y, lam):
    p = X.shape[1]
    nu = 1 / np.max(np.linalg.eigvals(X.T @ X))
    alpha = 1
    beta = np.zeros(p)
    beta_old = np.zeros(p)
    gamma = np.zeros(p)
    eps = 1
    while eps > 0.001:
        w = gamma + nu * X.T @ (y - X @ gamma)
        beta = max(1 - lam * nu / np.linalg.norm(w, 2), 0) * w
        alpha_old = copy.copy(alpha)
        alpha = (1 + np.sqrt(1 + 4 * alpha**2)) / 2
        gamma = beta + (alpha_old - 1) / alpha * (beta - beta_old)
        eps = np.max(np.abs(beta - beta_old))
        beta_old = copy.copy(beta)
    return beta

n = 100
p = 3
X = randn(n, p)
beta = randn(p)
epsilon = randn(n)
y = 0.1 * X @ beta + epsilon
lambda_seq = np.arange(1, 50, 0.5)
m = len(lambda_seq)
beta_est = np.zeros((m, p))
for i in range(m):
    est = fista(X, y, lambda_seq[i])
    beta_est[i, :] = est

plt.xlim(0, 50)
plt.ylim(np.min(beta_est), np.max(beta_est))
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["係数1", "係数2", "係数3"]
for i in range(p):
    plt.plot(lambda_seq, beta_est[:, i], label="{}".format(labels[i]))
plt.legend(loc="upper right")
plt.show()

n = 100
p = 8
X = randn(n, p)
beta = randn(p)
epsilon = randn(n)
y = 0.1 * X @ beta + epsilon
lambda_seq = np.arange(1, 50, 0.5)
m = len(lambda_seq)
beta_est = np.zeros((m, p))
for i in range(m):
    est = fista(X, y, lambda_seq[i])
    beta_est[i, :] = est

plt.xlim(0, 50)
plt.ylim(np.min(beta_est), np.max(beta_est))
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["係数1", "係数2", "係数3","係数4", "係数5", "係数6","係数7", "係数8"]
for i in range(p):
    plt.plot(lambda_seq, beta_est[:, i], label="{}".format(labels[i]))
plt.legend(loc="upper right")
plt.show()