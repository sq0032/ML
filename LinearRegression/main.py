'''
Created on 2014/3/12

@author: squall
'''

def costfunction(X, y, theta):
    m = float(X.shape[0])
    cost = (1./(2.*m))*(X*theta-y).T*(X*theta-y)
    return cost.flat[0]

def gradient(X, y, theta, iter, alpha):
    theta_iter = [] #record theta for each iteration
    cost_iter = []  #record cost for each iteration
    m = float(X.shape[0])

    for i in range(iter):
        #update theta
        theta = theta-(alpha/m)*X.T*(X*theta-y)
        theta_iter.append(theta)
        cost_iter.append(costfunction(X,y,theta))

    return(theta, theta_iter, cost_iter)

import matplotlib.pyplot as plt
import numpy as np

#load data
data = np.loadtxt('ex1data1.txt', delimiter=',')

#set initial variables
x = np.ones(data.shape)
x[:,1] = data[:,0]
y = np.zeros(shape=(data.shape[0],1))
y[:,0] = data[:,1]
theta = np.matrix([[0.],[0.]])
alpha = 0.003
iter = 3000

#gradient descent
theta, theta_iter, cost_iter = gradient(x, y, theta, iter, alpha)

#plot result
result = x*theta
plt.plot(data[:,0], result)
plt.scatter(data[:,0], data[:,1], marker='x', c='r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

#plot cost
plt.plot(np.arange(0, iter, 1), cost_iter)
plt.show()

print('done')