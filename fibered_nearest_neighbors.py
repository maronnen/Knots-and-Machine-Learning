'''
Implementation of K-nearest neighborhs model on data obtained from knotinfo.math.indina.edu

We take in some numerical features (which are knot invariants) and attempt to learn if a given knot is
fibered or not.

This is mostly just practice for myself, but the accuracy of the model is better than I expected.
'''

import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#read the .csv and look at the head
data = pd.read_csv('knotinfo.csv')
print(data.head)

print("Converting the Fibered column to consist of 1 for Y and 0 for N\n")

#one-hot encoding of 'Fibered'
# Y = 1, N = 0
data = data.replace(['Y'],1)
data = data.replace(['N'],0)
print(data.head)

print("Splitting the data into training and test sets.\n")

#get features
features = ['Crossing Number', 'Bridge Index', 'Braid Index', 'Determinant']
X = data[features]
print("The shape of X: {}\n".format(X.shape))

#get outputs
y = data['Fibered']
print("The shape of y: {}\n".format(y.shape))


#split the data into training and test sets
test_proportion = .3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state=0)
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of y_test: {}\n".format(y_test.shape))

#grr = pd.plotting.scatter_matrix(X_train, c=y_train, figsize=(15,15), marker='o',
	#hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
#plt.show()


#k nearest neighbors on training set
print("Fitting a K-nearest neighborhs model to the training set.\n")
num_of_neighbors = 3
knn = KNeighborsClassifier(n_neighbors=num_of_neighbors)
knn.fit(X_train, y_train)


#evaluate the model
print("Evaluating the model on the test set.\n")
y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))