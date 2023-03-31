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


X0 = pd.read_csv('研究室\第7章/number\mnist0.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X1 = pd.read_csv('研究室\第7章/number\mnist1.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X2 = pd.read_csv('研究室\第7章/number\mnist2.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X3 = pd.read_csv('研究室\第7章/number\mnist3.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X4 = pd.read_csv('研究室\第7章/number\mnist4.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X5 = pd.read_csv('研究室\第7章/number\mnist5.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X6 = pd.read_csv('研究室\第7章/number\mnist6.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X7 = pd.read_csv('研究室\第7章/number\mnist7.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X8 = pd.read_csv('研究室\第7章/number\mnist8.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
X9 = pd.read_csv('研究室\第7章/number\mnist9.csv',sep = "," , header= None, ).iloc[0:500].to_numpy()
n = 1
a = 0.01
samp = 100
V = [X0,X1,X2,X3,X4,X5,X6,X7,X8,X9]
def soft_th(lambd, x):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)
def SCoTLASS(lambd, X):
    p = X.shape[1]
    v = np.random.normal(size=p)
    v = v / np.linalg.norm(v, 2)
    for k in range(200):
        u = np.dot(X, v)
        u = u / np.linalg.norm(u, 2)
        v = np.dot(X.T, u)
        v = soft_th(lambd, v)
        size = np.linalg.norm(v, 2)
        if size > 0:
            v = v / size
        else:
            break
    if np.linalg.norm(v, 2) == 0:
        print("vの全要素が0になった")
    return v
for i in range(10):
   x = V[i]
   img = Image.fromarray(np.uint8(x[samp].reshape(28,28)))
   img.save('研究室\第7章/数字の結果/mnist' + str(i) + '.png')
   for p in range(500):
      x[p] = x[p] - np.mean(x,axis = 0)
   pca = PCA(n_components=n)
   pca.fit(x)
   values = pca.transform(x)
   data2 = pca.inverse_transform(values)
   img = Image.fromarray(np.uint8(data2[samp].reshape(28,28)))
   img.save('研究室\第7章\数字の結果/mnist' + str(i) + 'pri.png')
   spca = SparsePCA(n_components=n,
                     max_iter=500 ,ridge_alpha=0,alpha = a,)
   lam = spca.fit_transform(x)
   shu = spca.components_
   X =lam[samp,0]*shu[0,:] 
   # for nu in range(1,n):
   #    X =X +lam[samp,nu]* shu[nu,:]
   X = X.reshape(28,28)
   img = Image.fromarray(np.uint8(X))
   img.save('研究室\第7章\数字の結果/mnist' + str(i)  + 'sp.png')
   img = Image.fromarray(np.uint8(-X))
   img.save('研究室\第7章\数字の結果/mnist' + str(i)  + 'sp-.png')



# for i in range(10):
#    y = V[i]
#    z = copy.copy(y)
#    for j in range(500):
#       z[j] = z[j] -  np.mean(z,axis=0)
#    v = SCoTLASS(0.000001,z)
#    X = z[samp]
#    v = np.dot(X.T,v)*v
#    img = Image.fromarray(np.uint8(v.reshape(28,28)))
#    img.save('研究室\第7章/数字の結果/mnist' + str(i) + 'scot.png')


