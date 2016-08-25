# Introduction To The Data
# try to estimate the leaning rate of the Leaning Tower of Pisa using a linear regression and interpret its coefficient and statistics.
import pandas
import matplotlib.pyplot as plt

pisa = pandas.DataFrame({"year": range(1975, 1988),
                         "lean": [2.9642, 2.9644, 2.9656, 2.9667, 2.9673, 2.9688, 2.9696,
                                  2.9698, 2.9713, 2.9717, 2.9725, 2.9742, 2.9757]})

print(pisa)

# Create a scatter plot with year on the x-axis and lean on the y-axis.
plt.scatter(pisa["year"], pisa["lean"])
plt.show()

"""Findings
From the scatter plot, I visually see that a linear regression looks to be a good fit for the data.
"""



# Fit The Linear Model
# Statsmodels is a library which allows for rigorous statistical analysis in python.
# The class sm.OLS is used to fit linear models, standing for oridinary least squares.
"""
OLS() does not automatically add an intercept to our model.
We can add a column of 1's to add another coefficient to our model and
since the coefficient is multiplied by 1 we are given an intercept.
"""
import statsmodels.api as sm

y = pisa.lean # target
X = pisa.year  # features
X = sm.add_constant(X)  # add a column of 1's as the constant term

# OLS -- Ordinary Least Squares Fit
linear = sm.OLS(y, X)
# fit model
linearfit = linear.fit()

# Print the summary of the model
print(linearfit.summary())




# Define A Basic Linear Model
"""
Mathematically, a basic linear regression model is defined as yi=β0+β1xi+ei where
ei∼N(0,σ2) is the error term for each observation i where β0 is the intercept and β1 is the slope.
The residual for the prediction of observation i is ei=yi^−yi where yi^is the prediction.
As introduced previously, N(0,σ2) is a normal distribution with mean 0 and a variance σ2.
This means that the model assumes that the errors, known as residuals, between our prediction and observed values are normally distributed
and that the average error is 0. Estimated coefficients, those which are modeled, will be refered to as βi^
while βi is the true coefficient which we cannot calculated.
In the end, yi^=β0^+β1^xi is the model we will estimate.
"""
# Our predicted values of y
yhat = linearfit.predict(X)
print(yhat)

# Using linearfit with data X and y predict the residuals.
residuals = yhat-y
print(residuals)
# Residuals are computed by subtracting the observed values from the predicted values.




# Histogram Of Residuals
"""
By creating a histogram of our residuals we can visually accept or reject the assumption of normality of the errors.
If the histogram of residuals look similar to a bell curve then we will accept the assumption of normality.
"""
# Create a histogram with 5 bins of the residuals
import matplotlib.pyplot as plt
plt.hist(residuals, bins=5)
plt.show()

"""Findings
Our dataset only has 13 observations in it making it difficult to interpret such a plot.
Though the plot looks somewhat normal the largest bin only has 4 observations.
In this case we cannot reject the assumption that the residuals are normal.
"""




# Sum Of Squares

