import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

pkl_file = 'hw3-data/hw3-data/q1_dataset.pkl'
data = pickle.load(open(pkl_file, 'rb'))
data = np.array(data)
X = data[:, :2]
Y = data[:, -1]
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)
plt.savefig('q1_graph')

print("\nC=1: ")

clf = svm.SVC()
clf.fit(X,Y)
print(clf.fit(X,Y).score(X,Y))
print(clf.fit(X,Y).dual_coef_)
print(clf.fit(X,Y).intercept_)

difference_list_c1 = np.subtract(clf.predict(X), Y)

isError = False

for i in range(len(difference_list_c1)):
    if difference_list_c1[i] != 0:
        if not(isError):
            print("\nError instances:")
            isError = True
        print("    -" + str(X[i]))



print("\n\nC=3: ")
clf3 = svm.SVC(C=3)
clf3.fit(X,Y)
print(clf3.fit(X,Y).score(X,Y))
print(clf3.fit(X,Y).dual_coef_)
print(clf3.fit(X,Y).intercept_)


difference_list_c3 = np.subtract(clf3.predict(X), Y)

isError = False

for i in range(len(difference_list_c3)):
    if difference_list_c3[i] != 0:
        if not(isError):
            print("\nError instances:")
            isError = True
        print("    -" + str(X[i]))
