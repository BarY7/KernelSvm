#################################
# Your name: Bar Yaacovi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels


def get_points():
    X, y = make_blobs(n_samples=120, centers=2,
                      random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_model = svm.SVC(C=1000, kernel='linear')
    poly_model = svm.SVC(C=1000, kernel='poly')
    rbf_model = svm.SVC(C=1000, kernel='rbf')
    linear_model.fit(X_train, y_train)
    poly_model.fit(X_train, y_train)
    rbf_model.fit(X_train, y_train)
    create_plot(X_train, y_train, linear_model)
    plt.show()
    create_plot(X_train, y_train, poly_model)
    plt.show()
    create_plot(X_train, y_train, rbf_model)
    plt.show()


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_candidates = [10**i for i in range(-5, 6)]
    accuracy_list = [0 for i in range(-5, 6)]
    accuracy_list_train = [0 for i in range(-5, 6)]
    for i in range(len(C_candidates)):
        candidate = C_candidates[i]
        linear_model = svm.SVC(C=candidate, kernel='linear')
        linear_model.fit(X_train, y_train)
        accuracy_list[i] = cross_validate(X_val, y_val, linear_model)
        accuracy_list_train[i] = cross_validate(X_train, y_train, linear_model)
        # create_plot(X_train, y_train, linear_model,)
        # plt.title("C = {candidate}".format(candidate=candidate))
        # plt.show()
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.plot(C_candidates, accuracy_list, label="Error on Validation Set")
    plt.plot(C_candidates, accuracy_list_train,
             label="Error on Training Set")
    plt.legend()
    plt.show()


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_candidates = [10**i for i in range(-5, 6)]
    accuracy_list = [0 for i in range(-5, 6)]
    accuracy_list_train = [0 for i in range(-5, 6)]
    for i in range(len(gamma_candidates)):
        candidate = gamma_candidates[i]
        rbf_model = svm.SVC(C=10, gamma=candidate, kernel='rbf')
        rbf_model.fit(X_train, y_train)
        accuracy_list[i] = cross_validate(X_val, y_val, rbf_model)
        accuracy_list_train[i] = cross_validate(X_train, y_train, rbf_model)
        # create_plot(X_train, y_train, rbf_model,)
        # plt.title("gamma = {candidate}".format(candidate=candidate))
        # plt.show()
    plt.xscale('log')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.plot(gamma_candidates, accuracy_list, label="Error on Validation Set")
    plt.plot(gamma_candidates, accuracy_list_train,
             label="Error on Training Set")
    plt.legend()
    plt.show()

# return accuracy of model on validation set


def cross_validate(x_val, y_val, model):
    sum_right = 0
    for i in range(len(x_val)):
        x, y = x_val[i], y_val[i]
        predict_y = model.predict([x])
        if(predict_y == y):
            sum_right = sum_right + 1
    return sum_right/len(x_val)
