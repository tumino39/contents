import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

def linear(X,y):
    p = X.shape[1]
    x_bar = np.zeros(p)
    for j in range(p):
        x_bar[j] = np.mean(X[:,j])
    for j in range(p):
        X[:,j] = X[:, j] - x_bar[j]
    y_bar=np.mean(y)
    y = y - y_bar
    beta = np.dot(
        np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y)
    )
    beta_0 = y_bar - np.dot(x_bar,beta)
    return beta, beta_0

X = np.array([[1,2],[2,2],[3,4]])
y = np.array([1,2,3])
print(linear(X,y))