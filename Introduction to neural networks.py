# Neural Networks And Iris Flowers
"""
Many machine learning prediction problems are rooted in complex data and its non-linear relationships between features.
Neural networks are a class of models that can learn these non-linear interactions between variables.
"""
"""
The DataFrame class includes a hist() method which creates a histogram for every numeric column in that DataFrame.
The histograms are generated using Matplotlib and displayed using plt.show()
"""
# read in the dataframe
import pandas
from numpy.random import permutation
import numpy as np

iris = pandas.read_csv("iris.csv")

# Shuffled the rows in the dataframe
shuffled_rows = permutation(iris.index)
iris = iris.iloc[shuffled_rows] # Do not forget .iloc for dataframe

print(iris.head())

# There are 2 species
print(iris.species.unique())

# Visualize the data using the method hist() on our DataFrame iris
import matplotlib.pyplot as plt
iris.hist()
plt.show()



# Neurons
# So far we have talked about methods which do not allow for a large amount of non-linearity.
"""
Neural networks are very loosely inspired by the structure of neurons in the human brain.
These models are built by using a series of activation units, known as neurons, to make predictions of some outcome.
Neurons take in some input, apply a transformation function, and return an output.
"""
"""
We will use the popular sigmoid (logistic) activation function because it returns values between 0 and 1 and can be treated as probabilities.
Sigmoid Function: g(z) = 1/(1 + e ** −z)
Sigmoid Activation Function: hθ(x) = 1 / (1+e ** ((−θ**T)*x))= 1 / (1+e ** −(θ0*1+θ1* x1+θ2* x2)
"""
z = np.asarray([[9, 5, 4]])   # np.asarray: Convert the input to an array
y = np.asarray([[-1, 2, 4]])
# a = [[1,2,3]]  # a datatype is a list

# np.dot is used for matrix multiplication
# z is 1x3 and y is 1x3,  z * y.T is then 3x3, y.T is a 3x1
print(np.dot(z,y.T))  # numpy.dot(a, b, out=None): Dot product of two arrays.(multiplication)

# Variables to test sigmoid_activation
"""This bias unit (1) is similar in concept to the intercept in linear regression and
it will shift the activity of the neuron to one direction or the other."""
# Add a "one" column into the dataframe
iris["ones"] = np.ones(iris.shape[0]) # numpy.ones(shape, dtype=None, order='C'): Return a new array of given shape and type, filled with ones.
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values # X is a ndarray
y = (iris.species == 'Iris-versicolor').values.astype(int) # y is a ndarray, "==" return True/False, .values turn it to 1 or 0
# .astype change the data type of each element in the array to int

# The first observation
x0 = X[0]  # x0 is a ndarray

# Initialize thetas randomly
theta_init = np.random.normal(0,0.01,size=(5,1)) # Draw random samples from a normal (Gaussian) distribution.
# numpy.random.normal(loc=0.0, scale=1.0, size=None), loc means center of the distribution, scale means std dev of the distribution.
# size means output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
# Default is None, in which case a single value is returned.
# theta_init is a 2d array (list of arrays)

def sigmoid_activation2(x, theta):
    x = np.asarray(x[None, :]) # x = [1, 2, 3, 4, 5]; array to matrix x[:, None] (shape(5,1)), x[None, :] (shape(1,5))
    theta = np.asarray(theta) # theta = [[1], [2], [3], [4], [5]] (shape(5,1))
    return 1 / (1 + np.exp(-np.dot(x, theta)[0])) # np.dot is matrix mutiplication  # [0] gets the first item in the list

# Sigmoid Activation Function: hθ(x) = 1 / (1+e ** ((−θ**T)*x))= 1 / (1+e ** −(θ0*1+θ1* x1+θ2* x2)
# Write a function called sigmoid_activation with inputs x a feature vector and theta a parameter vector of the same length to implement the sigmoid activation function.
from math import exp

def sigmoid_activation(x, theta):
    pair_total = 0
    for index in range(0, len(x)):
        pair_total = pair_total + (theta[index] * x[index])

    # 2 is better
    two_method = 2
    if two_method == 1:
        result = 1 / (1 + exp(-pair_total))  # exp(x) = e ** x
        # math.exp will remove matrix [] format
        result = [result]
        # recover matrix [] format
    elif two_method == 2:
        result = 1 / (1 + np.exp(-pair_total))  # exp(x) = e ** x
        # np.exp can keep matrix [] format

    return result

# Assign the value of sigmoid_activation(x0, theta_init) to a1. a1 should be a vector.
a1 = sigmoid_activation(x0, theta_init)
print(a1)

"""Finding
a1 is a probability between 0 to 1. It means taking the values in the first rwo in to account,
the prob of a1 being Iris-versicolor specis is 0.477"""





# Cost Function
"""
We can train a single neuron as a two layer network using gradient descent.
As we learned in the previous mission, we need to minimize a cost function which measures the error in our model.
The cost function measures the difference between the desired output and actual output, defined as:
J(θ) = (−1/m) ∑i=1-m (yi*log(hθ(xi)) + (1−yi)log(1−hθ(xi)) )

Since our targets, yi, are binary, either yi or (1−yi) will equal zero.
One of the terms in the summation will disappear because of this result and the activation function is then used to compute the error.
For example, if we observe a true target, yi=1, then we want hθ(xi) to also be close to 1.
So as hθ(xi) approaches 1, the log(hθ(xi)) becomes very close to 0.
Since the log of a value between 0 and 1 is negative, we must take the negative of the entire summation to compute the cost.
The parameters are randomly initialized using a normal random variable with a small variance, less than 0.1.
"""
# First observation's features and target
x0 = X[0]
y0 = y[0]

# Initialize parameters, we have 5 units and just 1 layer
theta_init = np.random.normal(0,0.01,size=(5,1))

# Write a function, singlecost(), that can compute the cost from just a single observation.
# This function should use input features X, targets y, and parameters theta to compute the cost function.
# sigmoid_activation function will return hθ(xi)

# Both singlecost function and singlecost2 function return the same thing
def singlecost2(X, y, theta):
    # Compute activation
    h = sigmoid_activation(X.T, theta)
    # Take the negative average of target*log(activation) + (1-target) * log(1-activation)
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

first_cost2 = singlecost2(x0, y0, theta_init)
print(first_cost2)

def singlecost(x, y, theta):
    m = len(x)
    h = sigmoid_activation(x, theta)
    total = -np.mean(y * np.log(h) + (1-y) * np.log(1 - h))
    return total
# Assign the cost of variables x0, y, and theta_init to variable first_cost.
first_cost = singlecost(x0, y0, theta_init)
print(first_cost)




