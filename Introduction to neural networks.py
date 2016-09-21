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
# z is 1x3 and y is 1x3,  z * y.T is then 1x1, y.T is a 3x1
print(np.dot(z,y.T))  # numpy.dot(a, b, out=None): Dot product of two arrays.(multiplication)

# Variables to test sigmoid_activation
"""This bias unit (1) is similar in concept to the intercept in linear regression and
it will shift the activity of the neuron to one direction or the other."""
# Add a "one" column into the dataframe
iris["ones"] = np.ones(iris.shape[0]) # numpy.ones(shape, dtype=None, order='C'): Return a new array of given shape and type, filled with ones.
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values # X is a ndarray X: [[1, 4.9, 2.5, 3. 2.1]\n[1, 3.3, 5.4, 5.2, 3.2]\n...]
y = (iris.species == 'Iris-versicolor').values.astype(int) # y is a ndarray, "==" return True/False, .values turn it to 1 or 0
# .astype change the data type of each element in the array to int

# The first observation
x0 = X[0]  # x0 is a ndarray

# Initialize thetas randomly
theta_init = np.random.normal(0,0.01,size=(5,1)) # Draw random samples from a normal (Gaussian) distribution.
# numpy.random.normal(loc=0.0, scale=1.0, size=None), loc means center of the distribution, scale means std dev of the distribution.
# size means output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
# Default is None, in which case a single value is returned.
# theta_init is a 2d array (list of arrays), theta_init = [[-0.00002]\n[0.004935]\n...]

def sigmoid_activation(x, theta):
    x = np.asarray(x)
    # another way: x = np.asarray(x[None, :]) # x = [1, 2, 3, 4, 5]; array to matrix x[:, None] (shape(5,1)), x[None, :] (shape(1,5))
    theta = np.asarray(theta) # theta = [[1], [2], [3], [4], [5]] (shape(5,1))
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))
    # Another way: return 1 / (1 + np.exp(-np.dot(x, theta)[0]))  # np.dot is matrix mutiplication  # [0] gets the first item in the list

# Sigmoid Activation Function: hθ(x) = 1 / (1+e ** ((−θ**T)*x))= 1 / (1+e ** −(θ0*1+θ1* x1+θ2* x2)
# Write a function called sigmoid_activation with inputs x a feature vector and theta a parameter vector of the same length to implement the sigmoid activation function.
from math import exp

def sigmoid_activation2(x, theta):
    pair_total = 0
    for index in range(0, len(x)):
        pair_total = pair_total + (theta[index] * x[index]) # sigmoid_activation2 use "dot" and here we use "*"

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

a2 = sigmoid_activation2(x0, theta_init)
print(a2)  # a1 and a2 are the same result

"""Finding
a1 is a probability between 0 to 1. It means taking the values in the first rwo in to account,
the prob of a1 being Iris-versicolor specis is 0.526"""





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

# Initialize parameters, we have 5 units and just 1 layer (randomly assign theta)
theta_init = np.random.normal(0,0.01,size=(5,1))

# Write a function, singlecost(), that can compute the cost from just a single observation.
# This function should use input features X, targets y, and parameters theta to compute the cost function.
# sigmoid_activation function will return hθ(xi)

# Both singlecost function and singlecost2 function return the same thing
def singlecost(X, y, theta):
    # Compute activation
    h = sigmoid_activation(X.T, theta)
    # Take the negative average of target*log(activation) + (1-target) * log(1-activation)
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

first_cost2 = singlecost(x0, y0, theta_init)
print(first_cost2)

def singlecost2(x, y, theta):
    m = len(x)
    h = sigmoid_activation(x, theta)  # singlecost2 we use X.T and here we put only x as the parameter
    total = -np.mean(y * np.log(h) + (1-y) * np.log(1 - h))
    return total
# Assign the cost of variables x0, y0, and theta_init to variable first_cost.
first_cost = singlecost(x0, y0, theta_init)
print(first_cost)





# Compute The Gradients
"""
compute the partial derivatives of the cost function to get the gradients.
Calculating derivatives are more complicated in neural networks than in linear regression.
Here we must compute the overall error and then distribute that error to each parameter.
Compute the derivative using the chain rule.
(∂J/∂θj)=(∂J/∂h(Θ))*(∂h(Θ)/∂θj)

This rule may look complicated, but we can break it down.
The first part is computing the error between the target variable and prediction.
The second part then computes the sensitivity relative to each parameter.
In the end, the gradients are computed as:
δ=(yi−hΘ(xi))*hΘ(xi)*(1−hΘ(xi))*xi

(yi−hΘ(xi)) is a scalar and the error between our target and prediction.
hΘ(xi)*(1−hΘ(xi)) is also a scalar and the sensitivity of the activation function.
xi is the features for our observation i.
δ is then a vector of length 5, 4 features plus a bias unit, corresponding to the gradients.

To implement this, we compute δ for each observation, then average to get the average gradient.
The average gradient is then used to update the corresponding parameters.
"""

# Store the updates into this array
grads = np.zeros(theta_init.shape) # grads is a 5x1 ndarray [[0]\n[0]\n[0]\n[0]\n[0]]
# grads2 = np.zeros(theta_init.shape)

# Number of observations
n = X.shape[0]

# Compute the average gradients over each observation in X and corresponding target y with the initialized parameters theta_init.
for index in range(n):
    h = sigmoid_activation(X[index], theta_init)  # h is a ndarray
    delta = (y[index]-h) * (h) * (1-h) * X[index]  # delta is a ndarray
    grads += delta[:, None]/n  # 'numpy.ndarray' object has no attribute 'append'  # grads is a ndarray
print(grads)

"""Another way:
for j, obs in enumerate(X):
    # Compute activation
    h = sigmoid_activation(obs, theta_init)
    # Get delta
    delta = (y[j]-h) * h * (1-h) * obs
    # accumulate
    grads2 += delta[:,np.newaxis]/X.shape[0]
print(grads2)
"""




# Two Layer Network
# use gradient descent to learn the parameters and predict the species of iris flower given the 4 features.
"""
Gradient descent minimizes the cost function by adjusting the parameters accordingly.
Adjust the parameters by substracting the product of the gradients and the learning rate from the previous parameters.
Repeat until the cost function coverges or a maximum number of iterations is reached.
"""
"""
while (number_of_iterations < max_iterations and (prev_cost - cost) > convergence_thres ) {
    update paramaters
    get new cost
    repeat
}
"""

theta_init = np.random.normal(0, 0.01, size=(5, 1))

# set a learning rate
learning_rate = 0.1
# maximum number of iterations for gradient descent
maxepochs = 10000
# costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.0001


def learn(X, y, theta, learning_rate, maxepochs, convergence_thres):
    costs = []
    cost = singlecost(X, y, theta)  # compute initial cost
    costprev = cost + convergence_thres + 0.01  # set an inital costprev to past while loop
    counter = 0  # add a counter
    # Loop through until convergence
    for counter in range(maxepochs):
        grads = np.zeros(theta.shape)  # to create a list of list
        for j, obs in enumerate(X):
            h = sigmoid_activation(obs, theta)  # Compute activation
            delta = (y[j] - h) * h * (1 - h) * obs  # Get delta
            grads += delta[:, np.newaxis] / X.shape[0]  # accumulate

        # update parameters
        theta += grads * learning_rate # if use "-=", another curve will come up, try if += or -=
        counter += 1  # count
        costprev = cost  # store prev cost
        cost = singlecost(X, y, theta)  # compute new cost
        costs.append(cost)
        if np.abs(costprev - cost) < convergence_thres:
            break

    plt.plot(costs)
    plt.title("Convergence of the Cost Function")
    plt.ylabel("J($\Theta$)") # plt.title('alpha > beta') => “alpha > beta”; plt.title(r'$\alpha > \beta$') => “a>B”in latex
    plt.xlabel("Iteration")
    plt.show()
    return theta

theta = learn(X, y, theta_init, learning_rate, maxepochs, convergence_thres)



# Neural Network
"""
Neural networks are usually built using mulitple layers of neurons.
Adding more layers into the network allows you to learn more complex functions.
"""
"""
We have a 3 layer neural network with four input variables x1,x2,x3, and x4 and a bias unit.
Each variable and bias unit is then sent to four hidden units, a1(2),a2(2),a3(2), and a4(2).
The hidden units have different sets of parameters θ.
"""
"""
θi,k(j) represents the parameter of input unit k which transform the units in layer j to activation unit ai(j+1).
This layer is known as a hidden layer because the user does not directly interact with it by passing or retrieving data.
The third and final layer is the output, or prediction, of our model.
Similar to how each variable was sent to each neuron in the hidden layer,
the activation units in each neuron are then sent to each neuron on the next layer.
"""

theta0_init = np.random.normal(0,0.01,size=(5,4))
theta1_init = np.random.normal(0,0.01,size=(5,1))
print(theta0_init)
print(theta1_init)

# Write a function feedforward() that will take in an input X and two sets of parameters theta0 and theta1
# to compute the output hΘ(X).
def feedforward(X, theta0, theta1):
    # feedforward to the first layer
    a1 = sigmoid_activation(X.T, theta0).T
    # add a column of ones for bias term
    a1 = np.column_stack([np.ones(a1.shape[0]), a1]) # np.column_stack: Stack 1-D arrays as columns into a 2-D array.
    # activation units are then inputted to the output layer
    out = sigmoid_activation(a1.T, theta1)
    return out


# Assign the output to variable h using features X and parameters theta0_init and theta1_init.
h = feedforward(X, theta0_init, theta1_init)
print(h)




# Multiple Neural Network Cost Function
"""
The cost function to multiple layer neural networks is identical to the cost function we used in the last screen, but hΘ(xi) is more complicated.
"""
# Write a function multiplecost() which estimates the cost function.
def multiplecost(X, y, theta0, theta1):
    h = feedforward(X, theta0, theta1)
    # Take the negative average of target*log(activation) + (1-target) * log(1-activation)
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return cost
# Assign the cost to variable c
c = multiplecost(X, y, theta0_init, theta1_init)
print(c)





# Backpropagation
"""
Now that we have multiple layers of parameters to learn, we must implement a method called backpropagation.
We've already implemented forward propagation by feeding the data through each layer and returning an output.
Backpropagation focuses on updating parameters starting at the last layer and circling back through each layer, updating accordingly.
"""
# There is no δ1 since the first layer are the features and have no error.
# To make the code more modular, we have refactored our previous code as a class, allowing us to organize related attributes and methods.

# Use a class for this model, it's good practice and condenses the code
class NNet3:
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

    def _feedforward(self, X):
        # feedforward to the first layer
        l1 = sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2

    def _multiplecost(self, X, y):
        # feed through network
        l1, l2 = self._feedforward(X)
        # compute error
        inner = y * np.log(l2) + (1 - y) * np.log(1 - l2)
        # negative of average error
        return -np.mean(inner)

    def predict(self, X):
        _, y = self._feedforward(X)
        return y

    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0, 0.01, size=(ncols, self.hidden_layer))
        self.theta1 = np.random.normal(0, 0.01, size=(self.hidden_layer + 1, 1))

        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres + 1  # set an initial costprev to past while loop
        counter = 0  # initialize a counter

        # Loop through until convergence
        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self._feedforward(X)

            # Start Backpropagation
            # Compute gradients
            l2_delta = (y - l2) * l2 * (1 - l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1 - l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:, 1:] / nobs * self.learning_rate

            # Store costs and check for convergence
            counter += 1  # Count
            costprev = cost  # Store prev cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev - cost) < self.convergence_thres and counter > 500:
                break


# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001
# Number of hidden units
hidden_units = 4

# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
model.learn(X, y)

# Plot costs
plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()




# Splitting Data

# First 42 rows to X_train and y_train
X_train = X[:42]
y_train = y[:42]
# Last 18 rows to X_test and y_test
X_test = X[42:]
y_test = y[42:]



# Predicting Iris Flowers
"""
To benchmark how well a three layer neural network performs when predicting the species of iris flowers,
you will have to compute the AUC, area under the curve, score of the receiver operating characteristic.
The function NNet3 not only trains the model but also returns the predictions.
The method predict() will return a 2D matrix of probabilities.
Since there is only one target variable in this neural network, select the first row of this matrix,
which corresponds to the type of flower.
"""
# Train the neural network using X_test and y_test and model, which has been initialized with a set of parameters.
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
model.learn(X_train, y_train)
# Once training is complete, use the predict() function to return the probabilities of the flower matching the species Iris-versicolor.
y = model.predict(X_test)
yhat = model.predict(X_test)[0]
# Compute the AUC score, using roc_auc_score() and assign it to auc.
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, yhat)
print(auc)
print(y)
print(yhat)













