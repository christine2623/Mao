# Introduction
# reads the data into a Dataframe, and cleans up some messy values
import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
filtered_cars = cars[cars['horsepower'] != '?']
filtered_cars['horsepower'] = filtered_cars['horsepower'].astype('float')




# Bias And Variance
"""
Bias and variance make up the 2 observable sources of error in a model that we can indirectly control.

Bias describes error that results in bad assumptions about the learning algorithm.
For example, assuming that only one feature, like a car's weight, relates to a car's fuel efficiency will lead you to
fit a simple, univariate regression model that will result in high bias.
The error rate will be high since a car's fuel efficiency is affected by many other factors besides just its weight.

Variance describes error that occurs because of the variability of a model's predicted values.
If we were given a dataset with 1000 features on each car and used every single feature to train an incredibly complicated multivariate regression model,
we will have low bias but high variance.

In an ideal world, we want low bias and low variance but in reality, there's always a tradeoff.
"""





# Bias-Variance Tradeoff
# Between any 2 models, one will overfit more than the other one.
# Every process has some amount of inherent noise that's unobservable. Overfit models tend to capture the noise as well as the signal in a dataset.
"""
We can approximate the bias of a model by training a few different models from the same class (linear regression in this case)
using different features on the same dataset and calculating their error scores.
For regression, we can use mean absolute error, mean squared error, or R-squared.
"""
"""
We can calculate the variance of the predicted values for each model we train
and we'll observe an increase in variance as we build more complex, multivariate models.
"""
"""
While an extremely simple, univariate linear regression model will underfit,
an extremely complicated, multivariate linear regression model will overfit.
Depending on the problem you're working on, there's a happy middle ground that will help you construct reliable and useful predictive models.
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Create a function named train_and_test that: Takes in a list of column names as the sole parameter (cols),
def train_and_test(cols):
    # Trains a linear regression model using: the columns in cols as the features, and the mpg column as the target variable.
    lr = LinearRegression()
    # Please use [[]] on the input column so the number of rows and columns of matrices can be right
    linear_model = lr.fit(filtered_cars[cols], filtered_cars["mpg"])
    # Uses the trained model to make predictions using the same input it was trained on
    prediction = linear_model.predict(filtered_cars[cols])
    # Computes the variance of the predicted values and the mean squared error between the predicted values and the actual label (mpg column).
    filtered_cars["prediction"] = prediction
    variance = (np.sum((filtered_cars["prediction"] - (np.mean(filtered_cars["prediction"]))) ** 2))/filtered_cars.shape[0]
    # Another way: variance = np.var(prediction)
    mse = mean_squared_error(filtered_cars["mpg"], prediction)
    # Returns the mean squared error value followed by the variance (e.g. return(mse, variance)).
    return (mse, variance)

# Use the train_and_test function to train a model using only the cylinders column. Assign the resulting mean squared error value and variance to cyl_mse and cyl_var.
cols = ["cylinders"]
cyl_mse, cyl_var = train_and_test(cols)
print(cyl_mse, cyl_var)
# Use the train_and_test function to train a model using only the weight column. Assign the resulting mean squared error value and variance to weight_mse and weight_var.
weight_mse, weight_var = train_and_test(["weight"])
print(weight_mse, weight_var)






# Multivariate Models
# Use the train_and_test function to train linear regression models using the following columns as the features:
# columns: cylinders, displacement.
# MSE: two_mse, variance: two_var.
cols = ["cylinders", "displacement"]
two_mse, two_var = train_and_test(cols)
print(two_mse, two_var)

# columns: cylinders, displacement, horsepower.
# MSE: three_mse, variance: three_var.
cols = ["cylinders", "displacement", "horsepower"]
three_mse, three_var = train_and_test(cols)
print(three_mse, three_var)

# columns: cylinders, displacement, horsepower, weight.
# MSE: four_mse, variance: four_var.
cols = ["cylinders", "displacement", "horsepower", "weight"]
four_mse, four_var = train_and_test(cols)
print(four_mse, four_var)

# columns: cylinders, displacement, horsepower, weight, acceleration.
# MSE: five_mse, variance: five_var.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
five_mse, five_var = train_and_test(cols)
print(five_mse, five_var)

# columns: cylinders, displacement, horsepower, weight, acceleration, model year
# MSE: six_mse, variance: six_var.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration","model year"]
six_mse, six_var = train_and_test(cols)
print(six_mse, six_var)

# columns: cylinders, displacement, horsepower, weight, acceleration, model year, origin
# MSE: seven_mse, variance: seven_var.
# Use print statements or the variable inspector below to display each value.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration","model year", "origin"]
seven_mse, seven_var = train_and_test(cols)
print(seven_mse, seven_var)





# Cross Validation
"""
A good way to detect if your model is overfitting is to compare the in-sample error and the out-of-sample error,
or the training error with the test error.
So far, we calculated the in sample error by testing the model over the same data it was trained on.
To calculate the out-of-sample error, we need to test the data on a test set of data.
"""
"""
If we do not have a separate test dataset, we'll instead use cross validation.
"""
"""
If a model's cross validation error (out-of-sample error) is much higher than the in sample error,
then your data science senses should start to tingle.
This is the first line of defense against overfitting and is a clear indicator that the trained model doesn't generalize well outside of the training set.
"""
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
# Create a function named train_and_cross_val that: takes in a single parameter (list of column names)
def train_and_cross_val(cols):
    feature = filtered_cars[cols]
    target = filtered_cars["mpg"]

    # uses the KFold class to perform 10-fold validation using a random seed of 3 (we use this seed to answer check your code),
    kf = KFold(filtered_cars.shape[0], 10, shuffle=True, random_state=3)

    variance_list = []
    mse_list = []
    # Iterate through each fold
    for train_index, test_index in kf:
        X_train, X_test = feature.iloc[train_index], feature.iloc[test_index]
        Y_train, Y_test = target.iloc[train_index], target.iloc[test_index]

        # trains a linear regression model using the features specified in the parameter,
        lr = LinearRegression()
        # Please use [[]] on the input column so the number of rows and columns of matrices can be right
        linear_model = lr.fit(X_train, Y_train)
        prediction = linear_model.predict(X_test)
        # Computes the variance of the predicted values and the mean squared error between the predicted values and the actual label (mpg column).
        # calculates the overall, mean squared error across all folds and the overall, mean variance across all folds.
        mse = mean_squared_error(Y_test, prediction)
        variance = np.var(prediction)
        mse_list.append(mse)
        variance_list.append(variance)

    avg_mse = np.mean(mse_list)
    avg_variance = np.mean(variance_list)

    # returns the overall mean squared error value then the overall variance (e.g. return(avg_mse, avg_variance)).
    return (avg_mse, avg_variance)

# Use the train_and_cross_val function to train linear regression models using the following columns as the features:
# the cylinders and displacement columns. Assign the resulting mean squared error value to two_mse and the resulting variance value to two_var.
cols = ["cylinders", "displacement"]
two_mse, two_var = train_and_cross_val(cols)
print(two_mse, two_var)

# the cylinders, displacement, and horsepower columns.
# Assign the resulting mean squared error value to three_mse and the resulting variance value to three_var.
cols = ["cylinders", "displacement", "horsepower"]
three_mse, three_var = train_and_cross_val(cols)
print(three_mse, three_var)

# the cylinders, displacement, horsepower, and weight columns.
# Assign the resulting mean squared error value to four_mse and the resulting variance value to four_var.
cols = ["cylinders", "displacement", "horsepower", "weight"]
four_mse, four_var = train_and_cross_val(cols)
print(four_mse, four_var)

# the cylinders, displacement, horsepower, weight, acceleration columns.
# Assign the resulting mean squared error value to five_mse and the resulting variance value to five_var.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
five_mse, five_var = train_and_cross_val(cols)
print(five_mse, five_var)

# the cylinders, displacement, horsepower, weight, acceleration, and model year columns.
# Assign the resulting mean squared error value to six_mse and the resulting variance value to six_var.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration","model year"]
six_mse, six_var = train_and_cross_val(cols)
print(six_mse, six_var)

# the cylinders, displacement, horsepower, weight, acceleration, model year, and origin columns.
# Assign the resulting mean squared error value to seven_mse and the resulting variance value to seven_var.
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration","model year", "origin"]
seven_mse, seven_var = train_and_cross_val(cols)
print(seven_mse, seven_var)




# Plotting Cross-Validation Error Vs. Cross-Validation Variance
"""
During cross validation, the more features we added to the model, the lower the mean squared error got.
This is a good sign and indicates that the model generalizes well to new data it wasn't trained on.
As the mean squared error value went up, however, so did the variance of the predictions.
This is to be expected, since the models with lower squared error values had higher model complexity,
which tends to be more sensitive to small variations in input values (or high variance).
"""

# On the same Axes instance:
# Generate a scatter plot with the model's number of features on the x-axis and the model's overall, cross-validation mean squared error on the y-axis.
# Use red for the scatter dot color.
num_of_features = [2, 3, 4, 5, 6, 7]
mse = [two_mse, three_mse, four_mse, five_mse, six_mse, seven_mse]
plt.scatter(num_of_features, mse, c="red")
# Generate a scatter plot with the model's number of features on the x-axis and the model's overall, cross-validation variance on the y-axis.
# Use blue for the scatter dot color.
var = [two_var, three_var, four_var, five_var, six_var, seven_var]
plt.scatter(num_of_features, var, c="blue")
plt.show()


""" Findings:
While the higher order multivariate models overfit in relation to the lower order multivariate models,
the in-sample error and out-of-sample didn't deviate by much.
The best model was around 50% more accurate than the simplest model.
On the other hand, the overall variance increased around 25% as we increased the model complexity.
"""