import matplotlib
from matplotlib import pyplot as plt
import pandas
import numpy as np

val = pandas.read_csv("./Data7/Training_labels.csv",header=None).values

plt.figure(1)

print(len(np.where(val[:,0]==1)[0]))
print(len(np.where(val[:,0]==0)[0]))
#plt.scatter(val[:,11],val[:,11])
#[-10, 6], [2, 10], [-5, -5]
#plt.scatter([-10,2,-5], [6,10,-5])

#plt.show()
