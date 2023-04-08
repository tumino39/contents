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
from japanmap import pref_names, pref_code, picture
from pylab import rcParams

def clean(z):
    m = len(z)
    j = 1
    while (z[0] >= z[j] and j < m-1):
        j = j + 1
    k = m - 2
    while (z[m-1] <= z[k] and k > 0):
        k = k - 1
    if j > k:
        return z[[0, m-1]]
    else:
        z_append = np.append(z[j:(k+1)], z[m-1])
        return np.append(z[0], z_append)

def G(i, theta, L, U, lam):
    if i == 0:
        return theta - y[0]
    elif (theta > L[i-1] and theta < U[i-1]):
        return G(i-1, theta, L, U, lam) + theta - y[i]
    elif theta >= U[i-1]:
        return lam + theta - y[i]
    else:
        return -lam + theta - y[i]

def fused(y, lam):
    if lam == 0:
        return y
    n = len(y)
    L = np.zeros(n-1)
    U = np.zeros(n-1)
    theta = np.zeros(n)
    L[0] = y[0] - lam
    U[0] = y[0] + lam
    z = [L[0], U[0]]
    if n > 2:
        for i in range(1, n-1):
            z = np.append(y[i] - 2*lam, z)
            z = np.append(z, y[i] + 2*lam)
            z = clean(z)
            m = len(z)
            j = 0
            while G(i, z[j], L, U, lam) + lam <= 0:
                j = j + 1
            if j == 0:
                L[i] = z[m-1]
                j = 1
            else:
                L[i] = (z[j-1]
                        - ((z[j] - z[j-1]) * (G(i, z[j-1], L, U, lam) + lam)
                           / (-G(i, z[j-1], L, U, lam)
                              + G(i, z[j], L, U, lam))))
            k = m - 1
            while G(i, z[k], L, U, lam) - lam >= 0:
                k = k - 1
            if k == m - 1:
                U[i] = z[0]
                k = m - 2
            else:
                U[i] = (z[k]
                        - ((z[k+1] - z[k]) * (G(i, z[k], L, U, lam) - lam)
                           / (-G(i, z[k], L, U, lam)
                              + G(i, z[k+1], L, U, lam))))
            z = z[j:(k+1)]
            z = np.append(L[i], z)
            z = np.append(z, U[i])
        z = np.append(y[n-1] - lam, z)
        z = np.append(z, y[n-1] + lam)
        z = clean(z)
        m = len(z)
        j = 0
    while (G(n-1, z[j], L, U, lam) <= 0 and j < m-1):
        j = j + 1
    if j == 0:
        theta[n-1] = z[0]
    else:
        theta[n-1] = (z[j-1]
                      - ((z[j] - z[j-1]) * G(n-1, z[j-1], L, U, lam)
                         / (-G(n-1, z[j-1], L, U, lam)
                            + G(n-1, z[j], L, U, lam))))
    for i in range(n-1, 0, -1):
        if theta[i] < L[i-1]:
            theta[i-1] = L[i-1]
        elif theta[i] > U[i-1]:
            theta[i-1] = U[i-1]
        else:
            theta[i-1] = theta[i]
    return theta

# # 実行に際しては，本章で定義される関数fusedを用いる
# df = np.loadtxt("第4章\cgh.txt", delimiter="\t")
# y = df
# n = len(y)
# lam = 1
# soln = fused(y, lam)
# plt.xlabel("遺伝子番号")
# plt.ylabel("コピー数比（対数値）")
# plt.scatter(np.arange(n), y, s=1)
# plt.plot(soln, color="red")
# plt.show()

# この関数を変えることで，地図の色を変えることができる
# def rgb(minimum, maximum, value):
#     minimum, maximum = float(minimum), float(maximum)
#     ratio = 2 * (value - minimum) / (maximum - minimum)
#     r = 255
#     g = int(max(0, 255 * (ratio / 2)))
#     b = 0
#     return r, g, b

def soft_th(lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def admm(y, D, lam):
    L, K = D.shape
    theta_old = np.zeros(K)
    theta = np.zeros(K)
    gamma = np.zeros(L)
    mu = np.zeros(L)
    rho = 1
    eps = 1
    while eps > 0.001:
        theta = (np.linalg.inv(np.eye(K) + rho * D.T @ D)
                 @ (y + D.T @ (rho * gamma - mu)))
        gamma = soft_th(lam, rho * D @ theta + mu) / rho
        mu = mu + rho * (D @ theta - gamma)
        eps = np.max(np.abs(theta - theta_old))
        theta_old = copy.copy(theta)
    return theta

# # 実行に際しては，本章で定義される関数admmを用いる
# lam = 50
# mat = np.loadtxt("第4章/adj.txt")
# y = np.loadtxt("第4章/2020_6_9.txt", delimiter=" ")
# u = []
# v = []
# for i in range(46):
#     for j in range(i+1, 47):
#         if mat[i, j] == 1:
#             u.append(i)
#             v.append(j)
# m = len(u)
# D = np.zeros((m, 47))
# for k in range(m):
#     D[k, u[k]] = 1
#     D[k, v[k]] = -1
# z = admm(y, D, lam)
# cc = np.round((10 - np.log(z)) * 2 - 1)
# min_cc = np.min(cc)
# max_cc = np.max(cc)
# data = {}
# for i in range(47):
#     data[i+1] = rgb(min_cc, max_cc, cc[i])

# rcParams["figure.figsize"] = 8, 8
# plt.imshow(picture())
# plt.imshow(picture(data))
# plt.show()

# n = 50
# x = np.array(range(n))
# y = np.sin(x / n * 2 * np.pi) + randn(n)
# lam = 1

def k_order(n, k):
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = 1
        D[i, i+1] = -1
    for j in range(1, k):
        DD = np.zeros((n-j-1, n-j))
        for i in range(n-j-2):
            DD[i, i] = 1
            DD[i, i+1] = -1
        D = np.dot(DD, D)
    return D

# 実行に際しては，本章で定義される関数admmを用いる
# k = 4
# D = k_order(n, k)
# soln = admm(y, D, lam)
# plt.xlabel("position")
# plt.ylabel("trend filtering estimate")
# plt.scatter(np.arange(n), y, s=1)
# plt.plot(soln, color="red")
# plt.show()

def lars(X, y):
    n, p = X.shape
    X_bar = np.zeros(p)
    for j in range(p):
        X_bar[j] = np.mean(X[:, j])
        X[:, j] = X[:, j] - X_bar[j]
    y_bar = np.mean(y)
    y = y - y_bar
    scale = np.zeros(p)
    for j in range(p):
        scale[j] = np.sqrt(np.sum(X[:, j] * X[:, j]) / n)
        X[:, j] = X[:, j] / scale[j]
    beta = np.zeros((p+1, p))
    lambda_seq = np.zeros(p+1)
    for j in range(p):
        lam = np.abs(np.sum(X[:, j] * y))
        if lam > lambda_seq[0]:
            j_max = j
            lambda_seq[0] = lam
    r = copy.copy(y)
    f_s = list(range(p))
    index = [j_max]
    Delta = np.zeros(p)
    for k in range(1, p):
        sub_s = list(set(f_s) - set(index))
        Delta[index] = (np.linalg.inv(X[:, index].T @ X[:, index])
                        @ X[:, index].T @ r / lambda_seq[k-1])
        u = X[:, sub_s].T @ (r - lambda_seq[k-1] * X @ Delta)
        v = -X[:, sub_s].T @ (X @ Delta)
        t = u / (v+1)
        for i in range(0, p-k):
            if t[i] > lambda_seq[k]:
                lambda_seq[k] = t[i]
                i_max = i
        t = u / (v-1)
        for i in range(0, p-k):
            if t[i] > lambda_seq[k]:
                lambda_seq[k] = t[i]
                i_max = i
        j = sub_s[i_max]
        index.append(j)
        beta[k, :] = beta[k-1, :] + (lambda_seq[k-1] - lambda_seq[k]) * Delta
        r = y - X @ beta[k, :]
    for k in range(p+1):
        for j in range(p):
            beta[k, j] = beta[k, j] / scale[j]
    return([beta, lambda_seq])


df = np.loadtxt("研究室\第4章\crime.txt", delimiter="\t")
X = df[:, 2:7]
y = df[:, 0]
res = lars(X, y)
beta, lambda_seq = res
p = beta.shape[1]


plt.xlim(0, 8000)
plt.ylim(-7.5, 15)
plt.xlabel(r"$\lambda$")
plt.ylabel("係数の値")
labels = ["警察への年間資金", "25歳以上で高校を卒業した人の割合",
          "16-19歳で高校に通っていない人の割合", "18-24歳で大学生の割合",
          "25歳以上で4年制大学を卒業した人の割合"]
cols = ["black", "red", "green", "blue", "cyan"]
for i in range(0, 5):
    plt.plot(lambda_seq[0:p], beta[0:p, i], color=cols[i],
             label="{}".format(labels[i]))
plt.legend(loc="upper right")
plt.show()
