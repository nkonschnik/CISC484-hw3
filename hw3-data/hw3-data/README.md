# README
## Prerequisites
`sklearn`

`matplotlib`

`numpy` (optional)
  
`pickle` (optional)


## Files
`q1_dataset.pkl`, `q1_dataset.txt`

`q2_training.pkl`, `q2_training.txt`

`q2_test.pkl`, `q2_test.txt`


Notes:
- The last column denotes the `label`. Other columns are the `attribute`/`feature`.
- Both `.txt` and `.pkl` provides the same contents. Use whichever you are more comfortable with.
`.pkl` is recommended for ease of programming.

### Usage
Visualize the dataset.
```
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

pkl_file = 'q1_dataset.pkl'
data = pickle.load(open(pkl_file, 'rb'))
data = np.array(data)
X = data[:, :2]
y = data[:, -1]
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.show()
```

## References
[1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

[2] https://scikit-learn.org/stable/modules/svm.html

## Hints
- Use `.support_vectors_` to access to **Support Vector**.
- Use `.dual_coef_` to access to **Dual coefficients** of the Support Vector in the decision function
multiplied by their *targets*, which equals \lambda * y in the lecture note notations.
