import copy
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

x = np.arange(-2,2,0.1)
y = x**2 -3 * x +np.abs(x)
plt.plot(x,y)
plt.scatter(1,-1,c = "red")
plt.title("$y = x^2-3x+|x|$")
plt.savefig("ex(1).png")   
plt.show()


x = np.arange(-2,2,0.1)
y = x**2 + x + 2 * np.abs(x)
plt.plot(x,y)
plt.scatter(0,0,c = "red")
plt.title("$y = x^2+x+2|x|$")
plt.savefig("ex(2).png")   
plt.show()