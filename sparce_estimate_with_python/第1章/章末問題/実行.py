import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

def soft_th(lam,x):
    return np.sign(x) * np.maximum(np.abs(x) - lam,0)

x = np.arange(-10,10,0.1)
y = soft_th(5,x)
plt.plot(x,y,c = "black")
plt.title(r"${\cal S}_\lambda(x)$",size = 24)
plt.plot([-5,-5],[-4,4],c = "blue",linestyle = "dashed")
plt.plot([5,5],[-4,4],c = "blue",linestyle = "dashed")
plt.text(-2,1,r"$\lambda = 5$",c ="red",size = 16)
plt.savefig("img.png")