import copy
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats

x = np.arange(-2, 2, 0.1)
y = 2*x**2 -  np.abs(x)
plt.plot(x, y)
plt.title("$y = x^2 - |x|$")
plt.show()
