# =========Machine Learning - Exercise 1: Linear Regression========
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

import numpy as np
import matplotlib.pyplot as plt
from numpy import mat

from computeCost import computeCost
from gradientDescent import gradientDescent

# ==================== Part 1: Basic Function ====================
print('5x5 Identity Matrix: \n');
A = np.eye(5)
print(A)

input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y) # number of training examples

# plot data
plt.scatter(X,y)
plt.xlabel('Population size in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

input('Program paused. Press enter to continue.\n')


# =================== Part 3: Cost and Gradient descent ===========
B = np.ones([m,1])
X = np.c_[B, X]  # Add a column of ones to x
y = np.reshape(y,(-1,1)) # reshape y to get (97,1)
theta = np.zeros([2,1])

# some gradient descent setting
iterarions = 1500
alpha = 0.01

print('\nTesting the cost function...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 32.07\n')

# compute and display initial cost

J = computeCost(X, y, np.array([[-1],[2]]) )
print('\nWith theta = [-1 ; 2]\nCost computed = %f \n' % J)
print('Expected cost value (approx) 54.24\n')

input('Program paused. Press enter to continue.\n')


print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, J_history = gradientDescent(X, y,theta, alpha, iterarions)

# print theta to screen
print('Theta found by gradient descent:\n')
print(theta,'\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

input('Program paused. Press enter to continue.\n')

# Plot the linear fit
plt.scatter(X[:,1], y)
plt.plot(X[:,1], np.dot(X, theta), 'r--')
plt.xlabel('Population size in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.reshape(np.array([1, 3.5]), (1,2)), theta)[0][0]
print('For population = 35,000, we predict a profit of %f\n' % (predict1*10000))
predict2 = np.dot(np.reshape(np.array([1, 7]), (1,2)), theta)[0][0]
print('For population = 70,000, we predict a profit of %f\n' % (predict2*10000))

input('Program paused. Press enter to continue.\n')

