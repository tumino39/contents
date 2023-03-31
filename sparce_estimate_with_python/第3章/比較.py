import copy
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats


n=100
p=300
X=randn(n*p).reshape(n,p)
beta = randn(p)
epsilon = randn(n)
y = 0.1*X@beta+epsilon
lam = 0.01
nu = 1/np.max(np.linalg.eigvals(X.T@X))
m = 50

#ISTAの性能評価
beta_est = np.zeros(p)
t = 0
val_I = np.zeros((m,p))
while t<m:
    val_I[t,] = beta_est
    beta_old =copy.copy(beta_est)
    gamma = beta_old + nu * X.T @ (y - X @ beta_old)
    beta_est = np.max(1-lam*nu/np.linalg.norm(gamma,2),0)*gamma
    t = t+1
vale = np.zeros(m)
val_final_I=val_I[m-1,:]
for i in range(m):
    vale[i] = np.linalg.norm(val_I[i,]-val_final_I)
plt.plot(vale,label="ISTA")


#FISTAの性能評価
beta_est = np.zeros(p)
beta_old = np.zeros(p)
gamma = np.zeros(p)
t = 0
val_F = np.zeros((m,p))
alpha = 1
while t<m:
    val_F[t,]=beta_est
    w = gamma + nu*X.T@(y-X@gamma)
    beta_est = np.max(1-lam*nu/np.linalg.norm(w,2),0)*w
    alpha_old = copy.copy(alpha)
    alpha = (1+np.sqrt(1+4*alpha**2))/2
    gamma = beta_est + (alpha_old-1)/alpha * (beta_est-beta_old)
    beta_old = copy.copy(beta_est)
    t  = t+1
vale = np.zeros(m)
val_final_F = val_F[m-1,:]
for i in range(m):
    vale[i]=np.linalg.norm(val_F[i,:]-val_final_F)
plt.plot(vale,label="FISTA")
plt.legend()
plt.title("ISTAとFISTA(p=300)")
plt.show()




