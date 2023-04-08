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
import copy as c
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler


# データ生成
n = 100
p = 5
x = np.random.normal(size=n*p).reshape(-1, p)
np.savetxt("研究室\第7章\X.csv", x, delimiter=",")
g = np.ones(315).reshape(5,3,21)

lam_seq = np.arange(0,105,5)/1000



for k in range(5):
    for i in range(21):
        spca = SparsePCA(n_components=3,
                        max_iter=500 ,ridge_alpha=0,alpha = lam_seq[i],)
        lam = spca.fit_transform(x)
        shu = spca.components_
        cros01 = np.dot(shu[0,:],shu[1,:])
        cros02 = np.dot(shu[0,:],shu[2,:])
        cros12 = np.dot(shu[1,:],shu[2,:])
        g[k,0,i] = np.abs(cros01)
        g[k,1,i] = np.abs(cros02)
        g[k,2,i] = np.abs(cros12)


a = np.mean(g,axis = 0)

plt.plot(lam_seq,a[0,:],label = '第1,2主成分')
plt.plot(lam_seq,a[1,:],label = '第1,3主成分')
plt.plot(lam_seq,a[2,:],label = '第2,3主成分')
plt.legend(loc = "upper right")
plt.show()



