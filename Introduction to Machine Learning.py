# Introduction To Machine Learning
"""
Here are a few specific examples:

How do the properties of a house affect it's market value?
How does an applicant's application affect if they get into graduate school or not?

In the first problem, we're interested in trying to predict a specific, real valued number -- the market value of a house in dollars.
Whenever we're trying to predict a real valued number, the process is called regression.

In the second problem, we're interested in trying to predict a binary value -- acceptance or rejection into graduate school.
Whenever we're trying to predict a binary value, the process is called classification.
"""




# Introduction To The Data
"""
Since the file isn't formatted as a CSV file and instead uses a variable number of white spaces to delimit the columns,
you can't use read_csv to read into a DataFrame. You need to instead use the read_table method,
setting the delim_whitespace parameter to True so the file is parsed using the whitespace between values"""
import pandas as pd

mpg = pd.read_table("auto-mpg.data", delim_whitespace=True)

"""
The file doesn't contain the column names unfortunately so you'll have to extract the column names
from auto-mpg.names and specify them manually. The column names can be found in the Attribute Information section.
"""

# Read the dataset auto-mpg.data into a DataFrame named cars using the Pandas method read_table.
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
# Specify that you want the whitespace between values to be used as the delimiter.
# Use the column names provided in auto-mpg.names to set the column names for the cars Dataframe.
# Display the cars DataFrame using a print statement or by checking the variable inspector below the code box.
print(cars.head())




# Exploratory Data Analysis
"""
Using this dataset, we can work on a more narrow problem:
How does the number of cylinders, displacement, horsepower, weight, acceleration, and model year affect a car's fuel efficiency?
Perform some exploratory data analysis for a couple of the columns to see which one correlates best with fuel efficiency."""
# Create a grid of subplots containing 2 rows and 1 column.
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Top chart: Scatter plot with the weight column on the x-axis and the mpg column on the y-axis.
ax1.scatter(x=cars["weight"], y=cars["mpg"])
# Another way: cars.plot("weight", "mpg", kind='scatter', ax=ax1)

# Bottom chart: Scatter plot with the acceleration column on the x-axis and the mpg column on the y-axis.
ax2.scatter(x=cars["acceleration"], y=cars["mpg"])
plt.show()

"""Findings:
The scatter plots hint that there's a strong negative linear relationship between the weight and mpg columns
and a weak, positive linear relationship between the acceleration and mpg columns. """


# Linear Relationship
# try to quantify the relationship between weight and mpg
"""
A machine learning model is the equation that represents how the input is mapped to the output.
Said another way, machine learning is the process of determining the relationship between the independent variable(s) and the dependent variable.
In this case, the dependent variable is the fuel efficiency and the independent variables are the other columns in the dataset.
"""
"""
we'll focus on a family of machine learning models known as linear models. These models take the form of:
y=mx+b
The input is represented as x, transformed using the parameters m (slope) and b (intercept),
and the output is represented as y. We expect m to be a negative number since the relationship is a negative linear one.

The process of finding the equation that fits the data the best is called fitting.
We'll use the Python library scikit-learn library to handle fitting the model to the data.
"""




# Scikit-Learn
"""
To fit the model to the data, we'll use the machine learning library scikit-learn.
Scikit-learn is the most popular library for working with machine learning models for small to medium sized datasets.
Even when working with larger datasets that don't fit in memory,
scikit-learn is commonly used to prototype and explore machine learning models on a subset of the larger dataset.
"""
"""
Scikit-learn uses an object-oriented style, so each machine learning model must be instantiated before it can be fit to a dataset
(similar to creating a figure in Matplotlib before you plot values).
We'll be working with the LinearRegression class from sklearn.linear_model:
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

To fit a model to the data, we use the conveniently named fit method:
lr.fit(inputs, output)
where inputs is a n_rows by n_columns matrix and output is a n_rows by 1 matrix.
"""
# Single NumPy array (398 elements).
print(cars["weight"].values)
# NumPy matrix (398 rows by 1 column).
print(cars[["weight"]].values)

# Import the LinearRegression class from sklearn.linear_model.
from sklearn.linear_model import LinearRegression
# Instantiate a LinearRegression instance and assign to lr.
lr = LinearRegression()
# Use the fit method to fit a linear regression model using the weight column as the input and the mpg column as the output.
lr.fit(cars[["weight"]], cars[["mpg"]])




# Making Predictions
# Now that we have a trained linear regression model, we can use it to make predictions.
# this model takes in a weight value, in pounds, and outputs a fuel efficiency value, in miles per gallon.
# To use a model to make predictions, use the LinearRegression method predict.
# The predict method has a single required parameter, the n_samples by n_features input matrix and
# returns the predicted values as a n_samples by 1 matrix (really just a list).
"""
Making predictions on data used for training is the first step in the testing & evaluation process.
If the model can't do a good job of even capturing the structure of the trained data,
then we can't expect it to do a good job on data it wasn't trained on.
This is known as underfitting, since the model under performs on the data it was fit on.
"""
# Use the LinearRegression method predict to make predictions using the values from the weight column.
predictions = lr.predict(cars[["weight"]])
# Assign the resulting list of predictions to predictions.
# Display the first 5 elements in predictions and the first 5 elements in the mpg column to compare the predicted values with the actual values.
print(predictions[:5])
print(cars["mpg"].head())





# Plotting The Model
# Generate a scatter plot with weight on the x-axis and the mpg column on the y-axis. Specify that you want the dots in the scatter plot to be red.
# Generate a scatter plot with weight on the x-axis and the predicted values on the y-axis. Specify that you want the dots in the scatter plot to be blue.
plt.scatter(x=cars["weight"], y=cars["mpg"], color="red")
plt.scatter(x=cars["weight"], y=predictions, color="blue")
plt.show()




# Error Metrics
"""
The plot from the last step gave us a visual idea of how well the linear regression model performs.
To obtain a more quantitative understanding, we can calculate the model's error,
or the mismatch between a model's predictions and the actual values.
"""
"""
One commonly used error metric for regression is mean squared error, or MSE for short.
You calculate MSE by computing the squared error between each predicted value and the actual value
"""
"""Example:
sum = 0
for each data point:
    diff =  predicted_value - actual_value
    squared_diff = diff ** 2
    sum += squared_diff
mse = sum/n
"""
# We'll use the mean_squared_error function from scikit-learn to calculate MSE.
# sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
from sklearn.metrics import mean_squared_error

# Use the mean_squared_error function to calculate the MSE of the predicted values and assign to mse.
mse =mean_squared_error(cars["mpg"], predictions)
print(mse)
import numpy
print("standard deviation : ", + numpy.std(cars["mpg"]))





# Root Mean Squared Error
"""
Root mean squared error, or RMSE for short, is the square root of the MSE and does a better job of penalizing large error values.
In addition, the RMSE is easier to interpret since it's units are in the same dimension as the data.
"""
rmse = mse ** (1/2)
print(rmse)




# In this mission, we focused on regression, a class of machine learning techniques where the input and output values are continuous values.



""" Try to perform k-fold cross validation in linear regression:
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np
# Create a new instance of the KFold class with the following properties:
# n set to length of admissions,
# 5 folds,
# shuffle set to True,
# random seed set to 8 (so we can answer check using the same seed),
# assigned to the variable kf.
kf = KFold(cars.shape[0], 5, shuffle=True, random_state=8)

# Create a new instance of the LogisticRegression class and assign to lr
model = LinearRegression()

# Use the cross_val_score function to perform k-fold cross-validation:
# using the LogisticRegression instance lr,
# using the gpa column for training,
# using the actual_label column as the target column,
# returning an array of accuracy values (one value for each fold).
accuracies = cross_val_score(model, cars[["weight"]], cars["mpg"], scoring="accuracy", cv=kf) // ValueError: continuous is not supported

# compute the average accuracy
average_accuracy = np.mean(accuracies)

print(accuracies)
print(average_accuracy)

# You can only perform accuracy_scoring with classification variable, no continuous variable
"""