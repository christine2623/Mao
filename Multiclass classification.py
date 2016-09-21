# Introduction To The Data
"""
The dataset we will be working with contains information on various cars.
For each car we have information about the technical aspects of the vehicle such as the motor's displacement, the weight of the car, the miles per gallon,
and how fast the car accelerates.
Using this information we will predict the origin of the vehicle, either North America, Europe, or Asia.
"""
import pandas as pd
cars = pd.read_csv("auto.csv")
# Use the Series method unique to assign the unique elements in the column origin to unique_regions.
# Then use the print function to display unique_origins. unique_origins is a ndarray.
unique_origins = cars["origin"].unique()
print(unique_origins)



# Dummy Variables
# categorical variables have been represented in the dataset using integer values (like 0 and 1)
"""
Though the column year is a number, each year could have vastly different manufacturing numbers.
Since we don't have this information it is always safer to treat discrete value as categorical variables.
"""
"""
We must use dummy variables for columns containing categorical values.
Whenever we have more than 2 categories, we need to create more columns to represent the categories.
Since we have 5 different categories of cylinders, we could use 3, 4, 5, 6, and 8 to represent the different categories.
We can split the column into separate binary columns:

cyl_3 -- Does the car have 3 cylinders? 0 if False, 1 if True.
cyl_4 -- Does the car have 4 cylinders? 0 if False, 1 if True.
cyl_5 -- Does the car have 5 cylinders? 0 if False, 1 if True.
cyl_6 -- Does the car have 6 cylinders? 0 if False, 1 if True.
cyl_8 --Does the car have 8 cylinders? 0 if False, 1 if True.
"""
"""
We can use the Pandas function get_dummies to return a Dataframe containing binary columns from the values in the cylinders column.
In addition, if we set the prefix parameter to cyl, Pandas will pre-pend the column names to match the style we'd like.
dummy_df = pd.get_dummies(cars["cylinders"], prefix="cyl")
"""
"""
We then use the Pandas function concat to add the columns from this Dataframe back to cars
cars = pd.concat([cars, dummy_df], axis=1)
"""
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)
print(cars.head())

# Use the Pandas function get_dummies to create dummy values from the year column.
# Use the prefix attribute to prepend year to each of the resulting column names.
# Assign the resulting Dataframe to dummy_years.
dummy_years = pd.get_dummies(cars["year"], prefix="year")
# Use the Pandas method concat to concatenate the columns from dummy_years to cars.
cars = pd.concat([cars, dummy_years], axis=1)
# Use the Dataframe method drop to drop the years and cylinders columns from cars.
# Please remember to assign the df after drop back to itself!!!
cars = cars.drop("year", axis=1)
cars = cars.drop("cylinders", axis=1)
# Display the first 5 rows of the new cars Dataframe to confirm.
print(cars.head())




# Multiclass Classification
"""
When we have 3 or more categories, we call the problem a multiclass classification problem.
There are a few different methods of doing multiclass classification and in this mission,
we'll focus on the one-versus-all method.
"""
"""
The one-versus-all method is a technique where we choose a single category as the Positive case
and group the rest of the categories as the False case.
We're essentially splitting the problem into multiple binary classification problems.
For each observation, the model will then output the probability of belonging to each category.
"""
import numpy as np
# randomized the cars Dataframe
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]

# Split the shuffled_cars Dataframe into 2 Dataframes: train and test.
# Assign the first 70% of the shuffled_cars to train.
print(shuffled_cars.shape[0])
train = shuffled_cars.iloc[0:42]
# Assign the last 30% of the shuffled_cars to test.
test = shuffled_cars.iloc[43:60]




# Training A Multiclass Logistic Regression Model
"""
In the one-vs-all approach, we're essentially converting an n-class (in our case n is 3) classification problem
into n binary classification problems. For our case, we'll need to train 3 models:
1. A model where all cars built in North America are considered Positive (1) and
those built in Europe and Asia are considered Negative (0).
2. A model where all cars built in Europe are considered Positive (1) and
those built in North America and Asia are considered Negative (0).
3. A model where all cars built in Asia are labeled Positive (1) and
those built in North America and Europe are considered Negative (0).
Each of these models is a binary classification model that will return a probability between 0 and 1.
When we apply this model on new data, a probability value will be returned from each model (3 total).
For each observation, we choose the label corresponding to the model that predicted the highest probability.
"""
# sort the list numerically and alphabetically
unique_origins.sort()
print(unique_origins)

models = {}

# For each value in unique_origins, train a logistic regression model with the following parameters (Documentation):
# X: Dataframe containing just the cylinder & year binary columns.
# y: list (or Series) of Boolean values:
# True if observation's value for origin matches the current iterator variable.
# False if observation's value for origin doesn't match the current iterator variable.
# Add each model to the models dictionary with the following structure:
# key: origin value (1, 2, or 3),
# value: relevant LogistcRegression model instance.
from sklearn.linear_model import LogisticRegression

# Dataframe containing just the cylinder & year binary columns
feature = [f for f in train.columns if (f.startswith("cyl")) | (f.startswith("year"))]

for item in unique_origins:
    lr = LogisticRegression()
    X_train = train[feature]
    Y_train = train["origin"] == item
    logistic_model = lr.fit(X_train, Y_train)
    models[item] = logistic_model




# Testing The Models
# run our test dataset through the models and evaluate how well they performed.
# For each origin value from unique_origins:
# Use the LogisticRegression predict_proba function to return the 3 lists of predicted probabilities for the test set
# and add to the testing_probs Dataframe.
# create a dataframe with column name set
testing_probs = pd.DataFrame(columns=unique_origins)
for item in unique_origins:
    result_proba = models[item].predict_proba(test[feature])
    # When adding data to a dataframe, don't forget to give the row and column number to put the data
    testing_probs[item] = result_proba[:, 1]

print(testing_probs.head())




# Choose The Origin
# select the origin with the highest probability of classification for that observation.
"""
use the Dataframe method .idxmax() to return a Series where each value corresponds to the column
or where the maximum value occurs for that observation.
We need to make sure to set the axis paramater to 1 since we want to calculate the maximum value across columns.
"""
# Classify each observation in the test set using the testing_probs Dataframe.
predicted_origins = testing_probs.idxmax(axis=1)
# Assign the predicted origins to predicted_origins and use the print function to display it.
print(predicted_origins)