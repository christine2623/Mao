# K Fold Cross Validation
# In the previous mission, we learned about cross validation,
# a technique for testing a machine learning model's accuracy on new data that the model wasn't trained on.
"""
K-fold cross-validation works by:

splitting the full dataset into k equal length partitions,
selecting k-1 partitions as the training set and
selecting the remaining partition as the test set
training the model on the training set,
using the trained model to predict labels on the test set,
computing an error metric (e.g. simple accuracy) and setting aside the value for later,
repeating all of the above steps k-1 times, until each partition has been used as the test set for an iteration,
calculating the mean of the k error values.

Using 5 or 10 folds is common for k-fold cross-validation.
"""
"""
When working with large datasets, often only a few number of folds are used because of the time and cost it takes,
with the tradeoff that having more training examples helps improve the accuracy even with less folds.
"""




# Partititioning The Data
import pandas as pd
from sklearn.linear_model import LogisticRegression

admissions = pd.read_table("admissions.data", delim_whitespace=True)
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

print(admissions.head())

import numpy as np
from numpy.random import permutation
shuffled_index = permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
admissions = shuffled_admissions.reset_index()

# Partition the dataset into 5 folds and store each row's fold in a new integer column named fold
# Use df.ix[index_slice, col_name] to mass assign a specific value for col_name for all of the rows in the index_slice.
# You can use this to set a value for a new column in the Dataframe as well.
admissions.ix[:128, 'fold'] = 1
# Somehow "for" loop doesn't work! (why why why??)
admissions.ix[129:257, 'fold'] = 2
admissions.ix[258:386, 'fold'] = 3
admissions.ix[387:514, 'fold'] = 4
admissions.ix[515:644, 'fold'] = 5

# Ensure the column is set to integer type.
admissions["fold"] = admissions["fold"].astype('int')

print(admissions.head())
print(admissions.tail())




# First Iteration
# let's assign fold 1 as the test set and folds 2 to 5 as the training set.
# Then, train the model and use it to predict labels for the test set.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

train_iteration_one = admissions[admissions["fold"] != 1]
test_iteration_one = admissions[admissions["fold"] == 1]

logistic_model = model.fit(train_iteration_one[["gpa"]], train_iteration_one["actual_label"])
labels = model.predict(test_iteration_one[["gpa"]])
test_iteration_one["predicted_label"] = labels
match = test_iteration_one["predicted_label"] == test_iteration_one["actual_label"]
correct_match = test_iteration_one[match]
iteration_one_accuracy = correct_match.shape[0] / test_iteration_one.shape[0]
print(iteration_one_accuracy)





# Function For Training Models
# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]
# Write a function named train_and_test that takes in a Dataframe and a list of fold id values (1 to 5 in our case) and returns a list of accuracy values
def train_and_test(df, list):
    model = LogisticRegression()
    iteration_accuracy = []
    for fold in list:
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]
        model = model.fit(train[["gpa"]], train["actual_label"])
        labels = model.predict(test[["gpa"]])
        test["predicted_label"] = labels
        match = test["predicted_label"] == test["actual_label"]
        correct_match = test[match]
        iteration_accuracy.append(float(correct_match.shape[0] / test.shape[0]))
    return iteration_accuracy

# Use the train_and_test function to return the list of accuracy values for the admissions Dataframe and assign to accuracies
accuracies = train_and_test(admissions, fold_ids)
# Compute the average accuracy and assign to average_accuracy.
average_accuracy = sum(accuracies)/len(accuracies)
# Another way: average_accuracy = np.mean(accuracies)
# average_accuracy should be a float value while accuracies should be a list of float values (one float value per iteration).
# Use the variable inspector or the print function to display the values for accuracies and average_accuracy
print(accuracies)
print(average_accuracy)




# Sklearn
"""
In many cases, the resulting accuracy values don't differ much between a simpler, less time-intensive method like holdout validation
and a more robust but more time-intensive method like k-fold cross-validation.
As you use these and other cross validation techniques more often,
you should get a better sense of these tradeoffs and when to use which validation technique.
"""
"""
In addition, the computed accuracy values for each fold stayed within 61% and 63%, which is a healthy sign.
Wild variations in the accuracy values between folds is usually indicative of using too many folds (k value)
"""

"""
Similar to having to instantiate a LinearRegression or LogisticRegression object before you can train one of those models,
you need to instantiate a KFold class before you can perform k-fold cross-validation.
kf = KFold(n, n_folds, shuffle=False, random_state=None)

n is the number of observations in the dataset,
n_folds is the number of folds you want to use,
shuffle is used to toggle shuffling of the ordering of the observations in the dataset,
random_state is used to specify a seed value if shuffle is set to True.
"""
"""
If we're primarily only interested in accuracy and error metrics for each fold,
we can use the KFold class in conjunction with the cross_val_score function,
which will handle training and testing of the models in each fold.
cross_val_score(estimator, X, Y, scoring=None, cv=None)

estimator is a sklearn model that implements the fit method (e.g. instance of LinearRegression or LogisticRegression),
X is the list or 2D array containing the features you want to train on,
y is a list containing the values you want to predict (target column),
scoring is a string describing the scoring criteria (list of accepted values here).
cv describes the number of folds. Here are some examples of accepted values:
    an instance of the KFold class,
    an integer representing the number of folds.
"""

"""
Here's the general workflow for performing k-fold cross-validation using the classes we just described:

instantiate the model class you want to fit (e.g. LogisticRegression),
instantiate the KFold class and using the parameters to specify the k-fold cross-validation attributes you want,
use the cross_val_score function to return the scoring metric you're interested in.
"""
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

# Create a new instance of the KFold class with the following properties:
# n set to length of admissions,
# 5 folds,
# shuffle set to True,
# random seed set to 8 (so we can answer check using the same seed),
# assigned to the variable kf.
kf = KFold(admissions.shape[0], 5, shuffle=True, random_state=8)

# Create a new instance of the LogisticRegression class and assign to lr
lr = LogisticRegression()

# Use the cross_val_score function to perform k-fold cross-validation:
# using the LogisticRegression instance lr,
# using the gpa column for training,
# using the actual_label column as the target column,
# returning an array of accuracy values (one value for each fold).
accuracies = cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="accuracy", cv=kf)

# compute the average accuracy
average_accuracy = np.mean(accuracies)

print(accuracies)
print(average_accuracy)





# Interpretation
"""
Using 5-fold cross-validation, we achieved an average accuracy score of 64.4%,
which closely matches the 63.6% accuracy score we achieved using holdout validation.
When working with simple univariate models, often holdout validation is more than enough and the similar accuracy scores confirm this.
When you're using multiple features to train a model (multivariate models),
performing k-fold cross-validation can give you a better sense of the accuracy you should expect when you use the model
on data it wasn't trained on.
"""
