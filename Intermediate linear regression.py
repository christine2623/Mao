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
# Stats models is a library which allows for rigorous statistical analysis in python.
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
"""
Many of the statistical measures used to evaluate linear regression models rely on three sum of squares measures.
The three measures include Sum of Square Error (SSE), Regression Sum of Squares (RSS), and Total Sum of Squares (TSS).
In aggregate each of these measures explain the variance of the entire model. We define the measures as the following:

SSE = ∑i=1-n (yi−yi^)**2 = ∑i=1-n ei**2.
We see that SSE is the sum of all residuals giving us a measure between the model's prediction and the observed values.

RSS = ∑i=1-n (yi¯−yi^)**2 where yi¯=(1/n) ∑i=1-n (yi).
RSS measures the amount of explained variance which our model accounts for.
For instance, if we were to predict every observation as the mean of the observed values then our model would be useless
and RSS would be very low. A large RSS and small SSE can be an indicator of a strong model.

TSS = ∑i=1-n (yi−yi¯)**2. TSS measures the total amount of variation within the data.

With some algebra we can show that TSS = RSS + SSE.
Intuitively this makes sense, the total amount of variance in the data is captured by the variance explained by the model
plus the variance missed by the model.
"""
# Compute the RSS and TSS for our model, linearfit, using the formulas above.
import numpy as np

# sum the (predicted - observed) squared
SSE = np.sum((yhat-y.values)**2)

# sum the (mean - predicted) squared
RSS = np.sum((np.mean(y)-yhat)**2)

# sum SSE and RSS
TSS = SSE + RSS

print(SSE)
print(RSS)
print(TSS)




# R-Squared
# The coefficient of determination, also known as R-Squared, is a great measure of linear dependence.
# It is a single number which tells us what the percentage of variation in the target variable is explained by our model.
# R**2=1−(SSE/TSS) = (RSS/TSS)
"""
Intuitively we know that a low SSE and high RSS indicates a good fit.
This single measure tells us what percentage of the total variation of the data our model is accounting for.
Correspondingly, the R2 exists between 0 and 1.
"""
# Compute the R-Squared for our model, linearfit. Assign the R-squared to variable R2.
R2 = RSS/TSS
print(R2)

"""Findings
the R-Squared value is very high for our linear model, 0.988, accounting for 98.8% of the variation within the data
"""




# Coefficients Of The Linear Model
"""
Each βi in a linear model f(x)=β0+β1x is a coefficient.
Fortunately there are methods to find the confidence of the estimated coefficients.
"""
"""
The row year corresponds to the independent variable x while lean is the target variable.
The variable const represents the model's intercept.

The coefficient measures how much the dependent variable will change with a unit increase in the independent variable.
For instance, we see that the coefficient for year is 0.0009.
This means that on average the tower of pisa will lean 0.0009 meters per year.
"""
# Assuming no external forces on the tower, how many meters will the tower of pisa lean in 15 years?
#The models parameters
print("\n",linearfit.params)

delta = (linearfit.params[1]) * 15
# Another way: delta = linearfit.params["year"] * 15
print(delta)




# Variance Of Coefficients
"""
The variance of each of the coefficients is an important and powerful measure.
In our example the coefficient of year represents the number of meters the tower will lean each year.
The variance of this coefficient would then give us an interval of the expected movement for each year.

 The standard error is the square root of the estimated variance.
 The estimated variance for a single variable linear model is defined as:
 s**2(β1^) = ∑i=1-n (yi−yi^)**2 / ((n−2) ∑i=1-n (xi−x¯)**2) = SSE / ((n−2) ∑i=1-n(xi−x¯)**2) .
"""
"""
Analyzing the formula term by term we see that the numerator, SSE, represents the error within the model.
A small error in the model will then decrease the coefficient's variance.

The denominator term, ∑i=1-n (xi−x¯)**2, measures the amount of variance within x.
A large variance within the independent variable decreases the coefficient's variance.

The entire value is divided by (n-2) to normalize over the SSE terms while accounting for 2 degrees of freedom in the model.

Using this variance we will be able to compute t-statistics and confidence intervals regarding this β1.
"""
# Compute s**2(β1^) for linearfit

SSE = np.sum((y.values - yhat)**2)
# Compute variance in X
# pisa.year = pisa["year"]
xvar = np.sum((pisa["year"] - pisa.year.mean())**2)
# Compute variance in b1
s2b1 = SSE / ((y.shape[0] - 2) * xvar)
print(s2b1)




# T-Distribution
"""
A common test of statistical signficance is the student t-test.
The student t-test relies on the t-distribution, which is very similar to the normal distribution,
following the bell curve but with a lower peak.
"""
"""
The t-distribution accounts for the number of observations in our sample set while
the normal distribution assumes we have the entire population.
In general, the smaller the sample we have the less confidence we have in our estimates.
The t-distribution takes this into account by increasing the variance relative to the number of observations.
You will see that as the number of observations increases the t-distribution approaches the normal distributions.
"""
"""
The density functions of the t-distributions are used in signficance testing.
The probability density function (pdf) models the relative likelihood of a continous random variable.
The cumulative density function (cdf) models the probability of a random variable being less than or equal to a point.
The degrees of freedom (df) accounts for the number of observations in the sample.
In general the degrees of freedom will be equal to the number of observations minus 2.
Say we had a sample size with just 2 observations, we could fit a line through them perfectly and no error in the model.
To account for this we subtract 2 from the number of observations to compute the degrees of freedom.
"""
"""
Scipy has a functions in the library scipy.stats.t which
can be used to compute the pdf and cdf of the t-distribution for any number of degrees of freedom.
scipy.stats.t.pdf(x,df) is used to estimate the pdf at variable x with df degrees of freedom.
"""
from scipy.stats import t

# 100 values between -3 and 3
x = np.linspace(-3,3,100)

# Compute the pdf with 3 degrees of freedom
print(t.pdf(x=x, df=3))
plt.plot(t.pdf(x=x, df=3))
plt.show()




# Statistical Significance Of Coefficients
"""
To do significance testing we must first start by stating our hypothesis.
We want to test whether the lean of the tower depends on the year, ie. every year the tower leans a certain amount.
This is done by setting null and alternative hypotheses.

In our case we will say the null hypothesis is that the lean of the tower of pisa does not depend on the year,
meaning the coefficient will be equal to zero.
H0:β1=0

The alternative hypothesis would then follow as the lean of the tower depend on the year,
the coefficient is not equal to zero.
H1:β1≠0
"""
"""
Testing the null hypothesis is done by using the t-distribution. The t-statistic is defined as,
t = abs(β1^−0) / ((s**2(β1^))**(1/2))

This statistic measures how many standard deviations the expected coefficient is from 0.
If β1 is far from zero with a low variance then t will be very high.
We see from the pdf, a t-statistic far from zero will have a very low probability.
"""
# Using the formula above, compute the t-statistic of β1
tstat = abs(linearfit.params.year - 0) / ((s2b1)**(1/2))
# Another way: tstat = linearfit.params["year"] / np.sqrt(s2b1)
print(tstat)




# The P-Value
"""
Finally, now that we've computed the t-statistic we can test our coefficient.
Testing the coefficient is easy, we need to find the probability of β1 being different than 0 at some significance level.
Lets use the 95% significance level, meaning that we are 95% certian that β1 differs from zero.
This is done by using the t-distribution.
By computing the cumulative density at the given p-value and degrees of freedom we can retrieve a corresponding probability.

A two-sided test, one which test whether a coefficient is either less than 0 and greater than 0, should be used for linear regression coefficients.
For example, the number of meters per year which the tower leans could be either positive or negative and we must check both.
To test whether β1 is either positive or negative at the 95% confidence interval we look at the 2.5 and 97.5 percentiles in the distribution,
leaving a 95% confidence between the two.
Since the t-distribution is symmetrical around zero we can take the absolute value and test only at the 97.5 percentile (the positive side).

If probability is greater than 0.975 than we can reject the null hypothesis (H0) and say that the year significantly affects the lean of the tower.
"""
# At the 95% confidence interval for a two-sided t-test we must use a p-value of 0.975
pval = 0.975

# The degrees of freedom
df = pisa.shape[0] - 2

# The probability to test against
p = t.cdf(tstat, df=df)
print(p)
plt.plot(p)
plt.show()
# Do we accept β1>0?
if p > pval:
    beta1_test = True
else:
    beta1_test = False
# Another way: beta1_test = p > pval
print(beta1_test)

"""Conclusion:
R-squared is a very powerful measure but it is often over used.
A low R-squared value does not necessarily mean that there is no dependency between the variables.
For instance, if the r-square would equal 0 but there is certianly a relationship.
A high r-squared value does not necessarily mean that the model is good at predicting future events because
it does not account for the number of observations seen.

Student t-tests are great for multivariable regressions where we have many features.
The test can allow us to remove certian variables which do not help the model signficantly and keep those which do.
"""























