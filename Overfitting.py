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




