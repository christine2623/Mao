# Classification
"""
the fundamental goal of machine learning is to understand the relationship between the independent variable(s) and the dependent variable.
"""
"""
In the previous mission, we explored a supervised machine learning technique called linear regression.
Linear regression works well when the target column we're trying to predict, the dependent variable, is ordered and continuous.
If the target column instead contains discrete values, then linear regression isn't a good fit.
"""
"""
In classification, our target column has a finite set of possible values which represent different categories a row can belong to.
We use integers to represent the different categories so we can continue to use mathematical functions
to describe how the independent variables map to the dependent variable.
"""




# Introduction To The Data
import pandas as pd
import matplotlib.pyplot as plt

admissions = pd.read_table("admissions.data", delim_whitespace=True)
print(admissions.head())

plt.scatter(admissions["gpa"],admissions["admit"])
plt.show()




# Logistic Regression
"""
Recall that the admit column only contains the values 0 and 1 and are used to represent binary values
and the numbers themselves don't carry any weight.
When numbers are used to represent different options or categories, they are referred to as categorical values.
Classification focuses on estimating the relationship between the independent variables and the dependent, categorical variable.
"""
# In this mission, we'll focus on a classification technique called logistic regression.
"""
While a linear regression model outputs a real number as the label, a logistic regression model outputs a probability value.
In binary classification, if the probability value is larger than a certain threshold probability,
we assign the label for that row to 1 or 0 otherwise.
"""




# Logit Function
# we use the logit function, which is a version of the linear function that is adapted for classification.
# in logistic regression the output has to be a real value between 0 and 1
""" we plot the logit function to visualize its properties:
define the logit function using the NumPy exp function,
generate equally spaced values, between -6 and 6 to represent the x-axis,
calculate the y-axis values by feeding each value in x to the logit function,
creating a line plot to visualize x and y.
"""
import numpy as np


# Logit Function
def logit(x):
    # np.exp(x) raises x to the exponential power, ie e^x. e ~= 2.71828
    return np.exp(x) / (1 + np.exp(x))


# Generate 500 real values, evenly spaced, between -6 and 6.
x = np.linspace(-6, 6, 500, dtype=float)

# Transform each number in t using the logit function.
y = logit(x)

# Plot the resulting data.
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()




# Training A Logistic Regression Model
# Since we're only working with one feature, gpa, this is referred to as a univariate model.
"""Training a logistic regression model in scikit-learn is similar to training a linear regression model,
with the key difference that we use the LogisticRegression class instead of the LinearRegression class."""
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]], admissions["admit"])

# Import the LogisticRegression class and instantiate a model named logistic_model.
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
# Use the LogisticRegression method fit to fit the model to the data.
# Notice that the gpa column is [[]] but the admit column is only [], (A column-vector y was passed when a 1d array was expected)
logistic_model.fit(admissions[["gpa"]], admissions["admit"])




# Plotting Probabilities
# the output of a logistic regression model is the probability that the row should be labelled as True, or in our case 1.
"""
To return the predicted probability, use the predict_proba method.
The only required parameter for this method is the num_features by num_sample matrix of observations
we want scikit-learn to return predicted probabilities for.
"""
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
# Probability that the row belongs to label '0'.
print(pred_probs[:,0])
# Probabililty that the row belongs to label '1'.
print(pred_probs[:,1])
# Create and display a scatter plot using the Matplotlib scatter function where:
plt.scatter(admissions["gpa"], pred_probs[:,1])
# the x-axis is the values in the gpa column,
# the y-axis is the probability of being classified as label 1.
plt.show()




# Predict Labels
"""
the scatter plot suggests a linear relationship between the gpa values and the probability of being admitted.
This if because logistic regression is really just an adapted version of linear regression for classification problems.
Both logistic and linear regression are used to capture linear relationships
between the independent variables and the dependent variable.
"""
# Use the LogisticRegression method predict to return the predicted for each label in the training set.
# The parameter for the predict method matches that of the predict_proba method:
# X: rows of data to use for prediction.
# Assign the result to fitted_labels.
fitted_labels = logistic_model.predict(admissions[["gpa"]])
# Use the print function to display the first 10 values in fitted_labels.
print(fitted_labels[:10])
plt.scatter(admissions["gpa"], fitted_labels)
plt.show()

