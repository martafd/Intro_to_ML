import numpy as np
from sklearn.tree import DecisionTreeClassifier

matrix = np.random.normal(loc=1, scale=10, size=(1000, 50))
m = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)
matrix_norm = (matrix - m) / std

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])
Z_sum = np.sum(Z, axis=1) > 10
Z_index = np.nonzero(Z_sum)

m1 = np.eye(3)
m2 = np.eye(3)
m2_under_m1 = np.vstack((m1,m2))


X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
clf.feature_importances_
np.isnan(X)