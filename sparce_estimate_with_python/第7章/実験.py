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


x = pd.read_csv('研究室\第7章/number\mnist9.csv',sep = "," , header= None, skiprows=5000).iloc[0:1000].to_numpy()
n = 2
img = Image.fromarray(np.uint8(x[2].reshape(28,28)))
img.save('研究室\第7章/数字の結果/mnist9.png')
pca = PCA(n_components=n)
pca.fit(x)
values = pca.transform(x)
data2 = pca.inverse_transform(values)
img = Image.fromarray(np.uint8(data2[2].reshape(28,28)))
img.save('研究室\第7章\数字の結果/prime_mnist9.png')


spca = SparsePCA(n_components=n,
                     max_iter=500 ,ridge_alpha=0,alpha = 5,
                     )

lam = spca.fit_transform(x)
shu = spca.components_
X = -lam[0,0]*shu[0,:] 
for i in range(1,n):
   X = X - lam[0,i] * shu[i,:]
X = X.reshape(28,28)
img = Image.fromarray(np.uint8(X))
img.save('研究室\第7章\数字の結果/sp_mnist9.png')


