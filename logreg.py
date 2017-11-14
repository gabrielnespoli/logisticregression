import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


# 1. Write a function z(x) that takes a real value x and returns the sigmoid function evaluated in x.
def z(X):
    return 1.0 / (1.0 + np.exp(-X))


# 2. Write a function h(theta, X) that takes a real array theta of length m and a real matrix X of size n-by-m,
# and returns a vector of length n whose i-th entry is the sigmoid function evaluated at <theta, X[i]>
# (<.,.> denotes inner product).
# X vector of observation
# each row of x is the sigmoid function evalutaion in theta
def h(Theta,X):
    return np.array([z(np.dot(Theta, x)) for x in X])


# 3. Write a function gradient(theta, X, Y) that computes the gradient of the log-likelihood function of the
# logistic model on the points (X, Y), evaluated at theta. The inputs are: a real array theta of length m,
# a real array X of size n-by-m, and a binary array Y of length y (that is, each entry of Y is either 0 or 1).
# The output is a real array of the same length as theta.
def gradient (Theta, X, Y):
    pX = h(Theta, X)  # i.e. [h(x) for each row x of X]
    return np.dot((Y - pX), X)


# 4. Write a function logfit(X, Y, alpha, itr) that performs a gradient ascent to find the vector of
# parameters theta that maximize the log-likelihood on the points (X,Y). The inputs are a real matrix X
# of size n-by-m, a binary array Y of length n, the step-size alpha, and the maximum number of iterations itr.
def logfit(X, Y, alpha=1, itr=100):
    Theta = np.zeros(X.shape[1])
    for i in range(itr):
        Theta += alpha * gradient(Theta, X, Y)
    return Theta


def normalize(X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0))


# 5. Write a function tprfpr(P, Y). The function takes in input a vector of estimated probabilities P
# and a vector of labels Y, both of length n. It returns a tuple of two arrays (tpr, fpr) whose i-th
# entries are the true positive rate and false positive rate obtained when as threshold we use the i-th
# smallest element of P (i.e. we predict label 1 for all entries of P having value at least equal to the i-th
# smallest element of P). Note that the P given in input need not be sorted.
def tprfpr(P, Y):
    Ysort = Y[np.argsort(P)[::-1]]
    ys = np.sum(Y)
    tpr = np.cumsum(Ysort) / ys  # [0, 0, 1, 2, 2, 3,..]/18
    fpr = np.cumsum(1 - Ysort) / (len(Y) - ys)
    return tpr,fpr


# 6. Write a function auc(fpr, tpr) that takes in input the two vectors of true positive rate and
# false positive rate, and computes the Area Under the Curve.
def auc(fpr,tpr):
    return (np.diff(tpr) * (1 - fpr[:-1])).sum()


def output(x,y,test):
    th = logfit(x, y, 0.001, 50)
    # 1. the vector theta of parameter estimates obtained through the logistic regression
    np.set_printoptions(precision=2)
    print(th)

    # 2. the ROC curve obtained on the training dataset, on a file called ID-roc.png, in PNG format
    P = h(th, x)  # predicted probabilities
    tpr, fpr = tprfpr(P, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC Curve')
    plt.plot(fpr, tpr)
    plt.savefig("1743585-roc.png")

    # 3. the AUC value obtained on the training dataset
    print(auc(fpr, tpr))

    # 4. the scores predicted for the observations in the test dataset
    nrows, ncolumns = test.shape
    index = list(test.columns.values)
    test_x = test[index]
    test_x = np.c_[np.ones(nrows),test_x] # concatenate a first column of ones (intercept term)
    test_y = h(th,test_x)
    print(test_y)


def main():
    training, test = sys.argv[1:]

    training = pd.read_csv(training, header=0)
    test = pd.read_csv(test, header=0)

    training = training.dropna()
    test = test.dropna()

    training = training.reset_index(drop=True)
    test = test.reset_index(drop=True)

    nrows, ncolumns = training.shape
    index = list(training.columns.values)

    x = training[index[:-1]]
    x = np.c_[np.ones(nrows), x] # concatenate a first column of ones (intercept term)
    y = training[index[-1]]
    th = logfit(x, y, 0.001, 50)
    print(th)


    output(x,y,test)


if __name__ == "__main__":
    main()
