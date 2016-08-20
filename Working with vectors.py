# Vectors

import numpy as np
# a vector is a single row or a single column.
vector1 = np.asarray([4, 5, 7, 10])
vector2 = np.asarray([8, 6, 3, 2])
vector3 = np.asarray([10, 4, 6, -1])
# Add vector1 and vector2 and assign the result to vector1_2.
vector1_2 = vector1 + vector2
# Add vector3 and vector1 and assign the result to vector3_1.
vector3_1 = vector3 + vector1

print(vector1_2)
print(vector3_1)




# Vectors And Scalars
# We can also multiply vectors by single numbers, called scalars.
vector = np.asarray([4, -1, 7])
# Multiply vector by the scalar 7 and assign the result to vector_7.
vector_7 = 7*vector
# Divide vector by the scalar 8 and assign the result to vector_8.
vector_8 = vector / 8
print(vector_7)
print(vector_8)





# Plotting Vectors
# We can do this with the .quiver() method of matplotlib.pyplot
""" Example
import matplotlib.pyplot as plt
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
X -- this is the origin of the vector (x coordinate)
Y -- the y-coordinate origin of the vector
U -- The distance the vector moves on the x axis.
V -- the distance the vector moves on the y axis.
The first item in each array corresponds to the first vector, the second item corresponds to the second vector, and so on.
"""
import numpy as np
import matplotlib.pyplot as plt

# We're going to plot 2 vectors
# The first will start at origin 0,0 , then go over 1 and up 2.
# The second will start at origin 1,2 then go over 3 and up 2.
X = [0,1]
Y = [0,2]
U = [1,3]
V = [2,2]
# Actually make the plot.
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

# Set the x axis limits
plt.xlim([0,6])
# Set the y axis limits
plt.ylim([0,6])
# Show the plot.
plt.show()
# adds a vector that starts at 0,0, and goes over 4 and up 4
plt.quiver([0,1,0], [0,2,0], [1,3,4], [2,2,4], angles='xy', scale_units='xy', scale=1)
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()





# Vector Length
#  The length of any vector, no matter how many dimensions, is just the square root of the sum of all of its elements squared.
# We're going to plot 3 vectors
# The first will start at origin 0,0 , then go over 2 (this represents the bottom of the triangle)
# The second will start at origin 2,2, and go up 3 (this is the right side of the triangle)
# The third will start at origin 0,0, and go over 2 and up 3 (this is our vector, and is the hypotenuse of the triangle)
X = [0,2,0]
Y = [0,0,0]
U = [2,0,2]
V = [0,3,3]
# Actually make the plot.
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()
vector_length = (2**2 + 3**2) ** (1/2)




# Dot Product
# The dot product can tell us how much of one vector is pointing in the same direction as another vector.
"""
we calculate the dot product by taking the first element of a, multiplying it by the first element of b,
then adding that to the second element of a multiplied by the second element of b,
then adding that to the third element of a multiplied by the third element of b."""
"""
This gives us a number that indicates how much of the length of a is pointing in the same direction as b.
If you project a onto the vector b, then it indicates how much of a is "in" vector b.
When two vectors are at 90 degree angles, the dot product will be zero."""
"""
determining if vectors are orthogonal.
Two vectors are orthogonal if they are perpendicular (that is, at a 90 degree angle to each other),
and their dot product is zero."""
# These two vectors are orthogonal
X = [0,0]
Y = [0,0]
U = [1,-1]
V = [1,1]
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()

dot = 3*5+4*6+5*7+6*8



# Making Predictions
"""
Now, let's try predicting how many points NBA players scored in 2013 using how many field goals they attempted.
Our algorithm will be single variable linear regression.
Remember that a single variable linear regression takes the form y=mx+b.
y is the variable we want to predict, x is the value of the predictor variable, m is the coefficient (slope), annd b is an intercept term.

the slope is 1.26, and the intercept is -18.92.

using the slope and intercept, we want to make predictions on the nba dataframe."""
import pandas
nba = pandas.read_csv("nba_stats.csv")

# For each row in nba:
# Predict the pts column using the fga column.
# Use the variables slope and intercept (already loaded in) to complete the linear regression equation.
slope = 1.26
intercept = -18.92
predictions = slope * nba["fga"] + intercept





# Vector And Matrix Multiplication
"""
We can multiply vectors and matrices. This kind of multiplication can enable us to perform linear regression much faster and more efficiently.
When multiplying a vector and a matrix, it can be useful to think of multiplying a matrix by a one column matrix instead"""
"""
In this case, we're multiplying a 2x2 matrix (A) by 2x1 matrix (B). The inner numbers must match in order for multiplication to work. """
"""
We multiply the first item in the first row of A by the first item in the first column of B.
We then multiply the second item in the first row of A by the second item in the first column of B.
We then add these values together to get the item at the position 1,1 in the result matrix.

We multiply the first item in the second row of A by the first item in the first column of B.
We then multiply the second item in the second row of A by the second item in the first column of B.
We then add these values together to get the item at the position 1,2 in the result matrix.

The resulting matrix will always have the same number of rows as the first matrix being multiplied has rows,
and the same number of columns as the second matrix being multiplied has columns.
"""




# Multiplying A Matrix By A Vector
# y=m1x1 + m2x2 + m3x3 + b
"""
Luckily, there's a faster and better way to solve linear regression equations, among other things.
It's matrix multiplication, and it's a foundational block of a lot of machine learning."""

# We can perform matrix multiplication in python using the .dot() method of numpy
import numpy as np
# Set up the coefficients as a column vector
coefs = np.asarray([[3], [-1]])
# Setup the rows we're using to make predictions
rows = np.asarray([[2,1], [5,1], [-1,1]])

# We can use np.dot to do matrix multiplication.  This multiplies rows by coefficients -- the order is important.
np.dot(rows, coefs)

nba_coefs = np.asarray([[slope], [intercept]])
# numpy.vstack() -> Stack arrays in sequence vertically (row wise). Take a sequence of arrays and stack them vertically to make a single array.
# numpy.ones() -> Return a new array of given shape and type, filled with ones.
# The .T accesses the attribute T of the object, which happens to be a NumPy array. The T attribute is the transpose of the array,
nba_rows = np.vstack([nba["fga"], np.ones(nba.shape[0])]).T

# Multiply nba_rows by nba_coefs.
predictions = np.dot(nba_rows, nba_coefs)
# nba_rows contains two columns -- the first is the field goals attempted by each player in 2013,
# and the second is a constant 1 value that enables us to add in the intercept.




# Applying Matrix Multiplication
"""
We multiply a matrix by another matrix in many machine learning methods, including neural networks.
Just like with linear regression, it enables us to do multiple calculations much more quickly than we could otherwise."""
"""
Let's say we wanted to multiply two matrices.
First, the number of columns of the first matrix has to equal the number of rows of the second matrix.
The final matrix will have as many rows as the first matrix, and as many columns as the second matrix.
An easy way to think of this is in terms of matrix dimensions.
We can multiply a 3x2 (rows x columns) matrix by a 2x3 matrix, and the final result will be 3x3
"""
A = np.asarray([[5,2], [3,5], [6,5]])
B = np.asarray([[3,1], [4,2]])

C = np.dot(A, B)
print(C)



