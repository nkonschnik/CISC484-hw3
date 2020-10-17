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

fignum = 1

for name, penalty in (('case1',1),('case2',3)):
    
    print("\nC=" + str(penalty) + ": ")
    clf = svm.SVC(kernel = 'linear', C=penalty)
    clf.fit(X,Y)
    print(clf.fit(X,Y).score(X,Y))
    #print(clf.fit(X,Y).dual_coef_)
    #print(clf.fit(X,Y).intercept_)

    difference_list_c1 = np.subtract(clf.predict(X), Y)

    isError = False

    for i in range(len(difference_list_c1)):
        if difference_list_c1[i] != 0:
            if not(isError):
                print("\nError instances:")
                isError = True
            print("    -" + str(X[i]))

    print(clf.support_vectors_)
    
    # get the separating hyperplane
    w = clf.dual_coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.savefig("q1_plot")
"""
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
"""