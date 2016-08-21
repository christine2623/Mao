# Data Cleaning
# explore how the horsepower of a car affects it's fuel efficiency and practice using scikit-learn to fit the linear regression model.
import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)

# Remove all rows where the value for horsepower is ? and convert the horsepower column to a float.
filtered_cars = cars[cars["horsepower"] != "?"]
filtered_cars['horsepower'] = filtered_cars["horsepower"].astype(float)

print(filtered_cars.head())




# Data Exploration
# generate a scatter plot that visualizes the relation between the horsepower values and the mpg values.
# Top: generate a scatter plot with the horsepower column on the x-axis and the mpg column on the y-axis.
# Bottom: generate a scatter plot with the weight column on the x-axis and the mpg column on the y-xis
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.scatter(filtered_cars["horsepower"], filtered_cars["mpg"])
ax2.scatter(filtered_cars["weight"], filtered_cars["mpg"])
plt.show()




# Fitting A Model
# fit a linear regression model using the horsepower values to get a quantitive understanding of the relationship.
import sklearn
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(filtered_cars[["horsepower"]], filtered_cars[["mpg"]])




# Plotting The Predictions
# plotted the predicted values and the actual values on the same plot to visually understand the model's effectiveness.
predictions = lr.predict(filtered_cars[["horsepower"]])

plt.scatter(filtered_cars["horsepower"], predictions, color="red")
plt.scatter(filtered_cars["horsepower"], filtered_cars["mpg"], color="blue")
plt.show()




# Error Metrics
# To evaluate how well the model fits the data, you can compute the MSE and RMSE values for the model.
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(filtered_cars["mpg"],predictions)
rmse = mse ** (1/2)

print(mse)
print(rmse)


"""Findings:
The MSE for the model from the last mission was 18.78 while the RMSE was 4.33.
Here's a table comparing the approximate measures for both models:

        Weight	Horsepower
MSE	    18.78	23.94
RMSE	4.33	4.89

If we could only use one input to our model, we should definitely use the weight values to predict the fuel efficiency values
because of the lower MSE and RMSE values.
"""