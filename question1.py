import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

pkl_file = 'hw3-data/hw3-data/q1_dataset.pkl'
data = pickle.load(open(pkl_file, 'rb'))
data = np.array(data)
X = data[:, :2]
y = data[:, -1]
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.show()