import numpy as np


# J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y

def computeCost(X, y, theta):
    # Initialize some useful values
    m = len(y)  # number of training examples
    temp = (np.dot(X, theta) - y) ** 2
    J = sum(temp) / (2 * m)

    return J
