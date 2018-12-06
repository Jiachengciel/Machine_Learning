import numpy as np

from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    J_history = np.zeros([iterations, 1])
    for i in range (iterations):
        m = len(y)
        # using the matrix is faster
        temp = np.dot( X.T, (np.dot(X, theta) - y) ) * alpha/m
        theta = theta - temp
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history