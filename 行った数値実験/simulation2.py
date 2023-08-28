# %%
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
from scipy.stats import multivariate_t
import random
from sklearn.decomposition import PCA
from pca_def import conv_pca,sph_pca2

# %%
in_dot = np.zeros((100,6))
in_eig = np.zeros((100,6))
out_dot = np.zeros((100,6))
out_eig = np.zeros((100,6))
for dim in range(6):
    d = 2**(dim + 5)
    alpha1 = 2
    sig2 = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            sig2[i,j] = 1/(np.abs(i - j) + 1)
    sig11 = np.zeros(d)
    sig11[0:2] = np.array([d**alpha1,d**(0.8*alpha1)])
    sig11 = np.diag(sig11)
    sigma2 = sig11 + sig2
    lam11,vec11 = np.linalg.eigh(sigma2)
    sort2  = lam11.argsort()[::-1]
    lam11 = lam11[sort2]
    vec11 = (vec11.T[sort2]).T
    for it in range(100):
        X = np.random.multivariate_normal(np.zeros(d),sigma2,20).T
        x2 = np.random.multivariate_normal(np.ones(d),2*d * np.identity(d),2).T
        X_out = np.concatenate([X,x2],axis = 1)
        d,n = np.shape(X)

        lam_s,vec_s_t,vec_s = sph_pca2(X)
        lam_so,vec_so_t,vec_so = sph_pca2(X_out)
        Z1 = np.diag(lam11**(-1/2)) @ vec11.T @ X
        u = np.zeros(n)
        for i in range(n):
            u[i] = Z1[0,i]/np.sqrt(Z1[0,i]**2+ lam11[1]/lam11[0] * Z1[1,i]**2 )
        u_out = np.zeros(n + 2)
        u_out[:n] = u
        u_out[n:] = np.array([1,1])
        u /= np.linalg.norm(u)
        u_out /= np.linalg.norm(u_out)
        in_dot[it,dim] = np.dot(u,vec_s[:,0])
        out_dot[it,dim] = np.dot(u_out,vec_so[:,0])
        lame = 0
        for p in range(n):
            lame += (Z1[0,p]**2)/(Z1[0,p]**2 + (lam11[1]/lam11[0]) * Z1[1,p]**2)
        lame /= n
        print('finish' + str(dim) +' ' +  str(it))

# %%
plt.plot(np.arange(5,11),(np.abs(in_dot)).mean(axis = 0),c = "r",marker = ".",label = "異常値無し")
plt.plot(np.arange(5,11),(np.abs(out_dot)).mean(axis = 0),c = "b",marker = ".",label = "異常値有り")
plt.xlabel("$log_2 d$")
plt.title("実際の固有ベクトルと推定される固有ベクトルの内積")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# %%



