import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats
import pandas as pd

N = 10

sample = np.zeros((N,2))
sample[:,0] = np.random.random_sample(N)

for i in range(N):
    sample[i,1] = np.sin(2 * np.pi * sample[i,0]) + np.random.normal(loc   = 0,scale = 0.5)


sin = np.zeros((1000,2))
sin[:,0] = np.linspace(0,1,1000)

for i in range(1000):
    sin[i,1] = np.sin(2 * np.pi * sin[i,0])


# plt.scatter(sample[:,0],sample[:,1])
# plt.plot(sin[:,0],sin[:,1],c = "green")
# plt.show()


fig, axes = plt.subplots(5, 2)

for i in range(10):
    if i < 5:
        res = np.polyfit(sample[:,0], sample[:,1], i)
        y = np.poly1d(res)(sin[:,0])
        axes[i,0].plot(sin[:,0],y,c = "red")
        axes[i,0].plot(sin[:,0],sin[:,1],c = "green")
        axes[i,0].scatter(sample[:,0],sample[:,1])
        axes[i,0].set_ylim(-1.5, 1.5)
    else:
        res = np.polyfit(sample[:,0], sample[:,1], i)
        y = np.poly1d(res)(sin[:,0])
        axes[i-5,1].plot(sin[:,0],y,c = "red")
        axes[i-5,1].plot(sin[:,0],sin[:,1],c = "green")
        axes[i-5,1].scatter(sample[:,0],sample[:,1])
        axes[i-5,1].set_ylim(-1.5, 1.5)

plt.show()

