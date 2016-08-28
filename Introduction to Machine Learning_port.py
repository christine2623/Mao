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

from sklearn.externals import joblib

joblib.dump(lr, "model.pkl")

