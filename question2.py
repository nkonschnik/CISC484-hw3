import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
import time

pkl_file = 'hw3-data/hw3-data/q2_training.pkl'
data = pickle.load(open(pkl_file, 'rb'))
data = np.array(data)
X = data[:, :2]
Y = data[:, -1]

pkl_file_test = 'hw3-data/hw3-data/q2_test.pkl'
dataTest = pickle.load(open(pkl_file_test, 'rb'))
dataTest = np.array(dataTest)
XTest = dataTest[:, :2]
YTest = dataTest[:, -1]


print("Kernel = linear: ")

clfLinear = svm.SVC(C=10, kernel='linear')
linearFit = clfLinear.fit(X,Y)

differenceListLinear = np.subtract(linearFit.predict(XTest), YTest)


errorCounterLinear = 0
for i in range(len(differenceListLinear)):
    if differenceListLinear[i] != 0:
        errorCounterLinear += 1
print("    Number of misclassifications: " + str(errorCounterLinear))
print("    Accuracy " + str(linearFit.score(XTest,YTest)))


print("\n\nKernel = rbf: ")
clfRBF = svm.SVC(C=10, kernel='rbf')
RBFfit = clfRBF.fit(X,Y)

differenceListRBF = np.subtract(RBFfit.predict(XTest), YTest)

errorCounterRBF = 0
for i in range(len(differenceListRBF)):
    if differenceListRBF[i] != 0:
        errorCounterRBF += 1

print("    Number of misclassifications: " + str(errorCounterRBF))
print("    Accuracy " + str(RBFfit.score(XTest,YTest)))
