import numpy as np

def conv_pca(X):
    d = X.shape[0]
    n = X.shape[1]
    barX = np.dot(X.mean(axis = 1).reshape(d,1),np.ones(n).reshape(1,n))
    X1 = X - barX
    if n >d:
        S = (1/(n-1)) * X1 @ X1.T
        lam2,vec2 = np.linalg.eigh(S)
        sort2  = lam2.argsort()[::-1]
        lam2 = lam2[sort2]
        vec_2t = (vec2.T[sort2]).T
    else:
        S = (1/(n-1)) * X1.T @ X1
        lam2,vec2 = np.linalg.eigh(S)
        sort2  = lam2.argsort()[::-1]
        lam2 = lam2[sort2]
        vec2 = (vec2.T[sort2]).T
        vec_2t = np.zeros((d,n-1))
        for i in range(n-1):
            vec_2t[:,i] = np.dot((1/np.sqrt(lam2[i] * (n-1))) * X1,vec2[:,i])
    return(lam2,vec_2t)

def sph_pca(X):
    d = X.shape[0]
    n = X.shape[1]
    cenX = np.dot(np.median(X,axis=1).reshape(d,1),np.ones(n).reshape(1,n))
    X_s1 = X - cenX
    X_s = np.zeros((d,n))
    N = np.zeros(n)
    for i in range(n):
        X_s[:,i] =X_s1[:,i]/ np.linalg.norm(X_s1[:,i],ord = 2)
        N[i] = np.linalg.norm(X_s1[:,i])
    barX_s = np.dot(X_s.mean(axis = 1).reshape(d,1),np.ones(n).reshape(1,n))
    X3 = X_s - barX_s
    if n > d:
        S_s = (1/(n-1)) * X3 @ X3.T
        lam3,vec3 = np.linalg.eigh(S_s)
        sort3  = lam3.argsort()[::-1]
        lam3 = lam3[sort3]
        vec_3t = (vec3.T[sort3]).T
    else:
        S_s = (1/(n-1)) * X3.T @ X3
        lam3,vec3 = np.linalg.eigh(S_s)
        sort3  = lam3.argsort()[::-1]
        lam3 = lam3[sort3]
        vec3 = (vec3.T[sort3]).T
        vec_3t = np.zeros((d,n-1))
        for i in range(n-1):
            vec_3t[:,i] = np.dot((1/np.sqrt(lam3[i] * (n-1))) * X3,vec3[:,i])
    return(lam3,vec_3t)

def sph_pca2(X):
    d = X.shape[0]
    n = X.shape[1]
    for i in range(n):
        X[:,i] =X[:,i]/ np.linalg.norm(X[:,i],ord = 2)
    barX = np.dot(X.mean(axis = 1).reshape(d,1),np.ones(n).reshape(1,n))
    X3 = X 
    if n > d:
        S_s = (1/(n)) * X3 @ X3.T
        lam3,vec3 = np.linalg.eigh(S_s)
        sort3  = lam3.argsort()[::-1]
        lam3 = lam3[sort3]
        vec_3t = (vec3.T[sort3]).T
    else:
        S_s = (1/(n-1)) * X3.T @ X3
        lam3,vec3 = np.linalg.eigh(S_s)
        sort3  = lam3.argsort()[::-1]
        lam3 = lam3[sort3]
        vec3 = (vec3.T[sort3]).T
        vec_3t = np.zeros((d,n-1))
        for i in range(n-1):
            vec_3t[:,i] = np.dot((1/np.sqrt(lam3[i] * (n-1))) * X3,vec3[:,i])
    return(lam3,vec_3t,vec3)
