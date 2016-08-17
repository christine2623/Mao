import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 2, 3, 4, 5]
# Going by our formula, every y value at a position is the same as the x-value in the same position.
# We could write y = x, but let's write them all out to make this more clear.
y = [0, 1, 2, 3, 4, 5]

# As you can see, this is a straight line that passes through the points (0,0), (1,1), (2,2), and so on.
plt.plot(x, y)
plt.show()

# Let's try a slightly more ambitious line.
# What if we did y = x + 1?
# We'll make x an array now, so we can add 1 to every element more easily.
x = np.asarray([0, 1, 2, 3, 4, 5])
y = x + 1

# y is the same as x, but every element has 1 added to it.
print(y)

# This plot passes through (0,1), (1,2), and so on.
# It's the same line as before, but shifted up 1 on the y-axis.
plt.plot(x, y)
plt.show()
# By adding 1 to the line, we moved what's called the y-intercept -- where the line intersects with the y-axis.
# Moving the intercept can shift the whole line up (or down when we subtract).


# Plot the equation y=x−1, using the existing x variable.
x = np.asarray([0,1,2,3,4,5])
y = x - 1
plt.plot(x,y)
plt.show()

# Plot the equation y=x+10, using the existing x variable.
x = np.asarray([0,1,2,3,4,5])
y = x + 10
plt.plot(x,y)
plt.show()



# Working With Slope
# y=mx . If we set the slope, mm, equal to 2, we'll get what we want.
import matplotlib.pyplot as plt
import numpy as np

x = np.asarray([0, 1, 2, 3, 4, 5])
# Let's set the slope of the line to 2.
y = 2 * x

# See how this line is "steeper" than before?  The larger the slope is, the steeper the line becomes.
# On the flipside, fractional slopes will create a "shallower" line.
# Negative slopes will create a line where y values decrease as x values increase.
plt.plot(x, y)
plt.show()

# Plot the equation y=4x, using the existing x variable.
x = np.asarray([0, 1, 2, 3, 4, 5])
y = 4 * x
plt.plot(x, y)
plt.show()

# Plot the equation y=.5x, using the existing x variable.
x = np.asarray([0, 1, 2, 3, 4, 5])
y = 5 * x
plt.plot(x, y)
plt.show()

# Plot the equation y=−2x, using the existing x variable.
x = np.asarray([0, 1, 2, 3, 4, 5])
y = -2 * x
plt.plot(x, y)
plt.show()





# Starting Out With Linear Regression
"""
For instance, if I know that how much I pay for my steak is highly positively correlated to the size of the steak (in ounces),
I can create a formula that helps me predict how much I would be paying for my steak.
The way we do this is with linear regression. Linear regression gives us a formula.
If we plug in the value for one variable into this formula, we get the value for the other variable.
y = mx + b"""
"""
We'll calculate slope first -- the formula is cov(x,y)/σ2x, which is just the covariance of x and y divided by the variance of x."""
# Calculate the slope you would need to predict the "quality" column (y) using the "density" column (x).
# We can use the cov function to calculate covariance, and the .var() method on Pandas series to calculate variance.
from numpy import cov
import pandas as pd

wine_quality = pd.read_csv("wine_quality.csv")
# cov function will require a matrix at the end, or it will become wrong type
slope_density = cov(wine_quality["density"], wine_quality["quality"])[0,1]/wine_quality["density"].var()




# Finishing Linear Regression
"""
Now that we can calculate the slope for our linear regression line, we just need to calculate the intercept.
The intercept is just how much higher or lower the average y point is than our predicted value.
We can compute the intercept by taking the slope we calculated and doing this: y¯−mx¯.
So we just take the mean of the y values, and then subtract the slope times the mean of the x values from that.
Remember that we can calculate the mean by using the .mean() method."""

# This function will take in two columns of data, and return the slope of the linear regression line.
def calc_slope(x, y):
  return cov(x, y)[0, 1] / x.var()

# calculate the intercept.
# dataframe[col].mean() can be used to calculate mean
intercept_density = wine_quality["quality"].mean() - (calc_slope(wine_quality["density"],wine_quality["quality"])*(wine_quality["density"].mean()))





# Making Predictions
"""
For example, a wine with a density of .98 isn't in our dataset,
but we can make a prediction about what quality a reviewer would assign to a wine with this density."""
from numpy import cov

def calc_slope(x, y):
    return cov(x, y)[0, 1] / x.var()

# Calculate the intercept given the x column, y column, and the slope
def calc_intercept(x, y, slope):
    return y.mean() - (slope * x.mean())

slope = calc_slope(wine_quality["density"], wine_quality["quality"])
intercept = calc_intercept(wine_quality["density"], wine_quality["quality"], slope)

def predict_y(x):
    result = slope * x + intercept
    return result


predicted_quality = wine_quality["density"].apply(predict_y)




# Finding Error
# An easier way to perform linear regression, using a function from scipy, the linregress function.
# We can also compute the distance between each prediction and the actual value -- these distances are called residuals.
"""
If we add up the sum of the squared residuals, we can get a good error estimate for our line.
We have to add the squared residuals, because just like differences from the mean, the residuals add to 0 if they aren't squared."""

from scipy.stats import linregress

# We've seen the r_value before -- we'll get to what p_value and stderr_slope are soon -- for now, don't worry about them.
slope, intercept, r_value, p_value, stderr_slope = linregress(wine_quality["density"], wine_quality["quality"])

# As you can see, these are the same values we calculated (except for slight rounding differences)
print(slope)
print(intercept)

# With the slope and the intercept, calculate the predicted y
predicted_y = predict_y(wine_quality["density"])

# Subtract each predicted y value from the corresponding actual y value, square the difference,
# and add all the differences together.
difference_square = (predicted_y - wine_quality["quality"]) ** 2
# predicted_y = numpy.asarray([slope * x + intercept for x in wine_quality["density"]]) works too
rss = sum(difference_square)

print(rss)




# Standard Error
"""
Standard error tries to make an estimate for the whole population of y-values --
even the ones we haven't seen yet that we may want to predict in the future.
The standard error lets us quickly determine how good or bad a linear model is at prediction.
The equation for standard error is (RSS/(n−2))** (1/2).
You take the sum of squared residuals, divide by the number of y-points minus two, and then take the square root."""

from scipy.stats import linregress
import numpy as np

# We can do our linear regression
# Sadly, the stderr_slope isn't the standard error, but it is the standard error of the slope fitting only
# We'll need to calculate the standard error of the equation ourselves
slope, intercept, r_value, p_value, stderr_slope = linregress(wine_quality["density"], wine_quality["quality"])

predicted_y = np.asarray([slope * x + intercept for x in wine_quality["density"]])
residuals = (wine_quality["quality"] - predicted_y) ** 2
rss = sum(residuals)

# Calculate the standard error using the above formula.
std_err = (rss/(len(wine_quality["quality"])-2))**(1/2)

# Calculate what percentage of actual y values are within 1 standard error of the predicted y value. Assign the result to within_one.
# Calculate what percentage of actual y values are within 2 standard errors of the predicted y value. Assign the result to within_two.
# Calculate what percentage of actual y values are within 3 standard errors of the predicted y value. Assign the result to within_three.
# Assume that "within" means "up to and including", so be sure to count values that are exactly 1, 2, or 3 standard errors away.
# For practice, one way to create std_dev is "std_predicted_y = predicted_y.std()"
within_1 = 0
within_2 = 0
within_3 = 0
for index in range(0, len(wine_quality["quality"])):
    if abs(wine_quality["quality"][index] - predicted_y[index]) <= std_err:
        within_1 += 1
    if abs(wine_quality["quality"][index] - predicted_y[index]) <= (2*std_err):
        within_2 += 1
    if abs(wine_quality["quality"][index] - predicted_y[index]) <= (3*std_err):
        within_3 += 1

within_one = within_1/len(wine_quality["quality"])
within_two = within_2/len(wine_quality["quality"])
within_three = within_3/len(wine_quality["quality"])

print(within_one)
print(within_two)
print(within_three)

