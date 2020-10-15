import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

pkl_file = 'hw3-data/hw3-data/q2_training.pkl'
data = pickle.load(open(pkl_file, 'rb'))
data = np.array(data)
X = data[:, :2]
Y = data[:, -1]

pkl_file_test = 'hw3-data/hw3-data/q2_test.pkl'
dataTest = pickle.load(open(pkl_file, 'rb'))
dataTest = np.array(data)
XTest = data[:, :2]
YTest = data[:, -1]


print("\nKernel = linear: ")

clfLinear = svm.SVC(C=10, kernel='linear')
clfLinear.fit(X,Y)
print("Accuracy " + str(clfLinear.fit(X,Y).score(X,Y)))

differenceListLinear = np.subtract(clfLinear.fit(X,Y).predict(XTest), YTest)


errorCounterLinear = 0
for i in range(len(differenceListLinear)):
    if differenceListLinear[i] != 0:
        errorCounterLinear += 1
print("Number of misclassifications: " + str(errorCounterLinear))


print("\n\nKernel = rbf: ")
clfRBF = svm.SVC(C=10, kernel='rbf')
clfRBF.fit(X,Y)
print("Accuracy " + str(clfRBF.fit(X,Y).score(X,Y)))


differenceListRBF = np.subtract(clfRBF.fit(X,Y).predict(XTest), YTest)


errorCounterRBF = 0
for i in range(len(differenceListRBF)):
    if differenceListRBF[i] != 0:
        errorCounterRBF += 1

print("Number of misclassifications: " + str(errorCounterRBF))