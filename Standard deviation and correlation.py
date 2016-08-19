# The Mean As The Center
"""
Interesting property about the mean:
If we subtract the mean of a set of numbers from each of the numbers, the differences will always add up to zero.
This is because the mean is the "center" of the data.
"""

# Make a list of values
import numpy

values = [2, 4, 5, -1, 0, 10, 8, 9]
# Compute the mean of the values
values_mean = sum(values) / len(values)
# Find the difference between each of the values and the mean by subtracting the mean from each value.
differences = [i - values_mean for i in values]
# This equals 0.  Try changing the values around and verifying that it equals 0 if you want.
print(sum(differences))

# We can use the median function from numpy to find the median
# The median is the "middle" value in a set of values -- if you sort the values in order, it's the one in the center
# (or the average of the two in the center if there are an even number of items in the set)
# You'll see that the differences from the median don't always add to 0.
from numpy import median
# numpy.median() to get the median of the list
values_median = numpy.median(values)
# calculate the difference between each element and median
differences2 = [i - values_median for i in values]
# add up all the median and print it
median_difference_sum = sum(differences2)
print(median_difference_sum)



# Finding Variance
# Variance tells us how "spread out" the data is around the mean.
"""
We looked at kurtosis earlier, which measures the shape of a distribution.
Variance directly measures how far from the mean the average element in the data is.
We calculate variance by subtracting every value from the mean, squaring the results, and averaging them.
"""
# The "pf" column in the data is the total number of personal fouls each player had called on them in the season --
# let's look at its variance.


import matplotlib.pyplot as plt
import pandas as pd

# The nba data is loaded into the nba_stats variable.
nba_stats = pd.read_csv("nba_stats.csv")

# Find the mean value of the column
pf_mean = nba_stats["pf"].mean()
# Initialize variance at zero
variance = 0
# Loop through each item in the "pf" column
for p in nba_stats["pf"]:
    # Calculate the difference between the mean and the value
    difference = p - pf_mean
    # Square the difference -- this ensures that the result isn't negative
    # If we didn't square the difference, the total variance would be zero
    # ** in python means "raise whatever comes before this to the power of whatever number is after this"
    square_difference = difference ** 2
    # Add the difference to the total
    variance += square_difference
# Average the total to find the final variance.
variance = variance / len(nba_stats["pf"])
print(variance)

# Compute the variance of the "pts" column in the data
# list.mean() will do the work
mean_pts = nba_stats["pts"].mean()
# set the variance to 0
variance_pts = 0
# for points for each player
for player in nba_stats["pts"]:
    # calculate the difference
    difference = player - mean_pts
    # get the square value
    square_difference = difference ** 2
    # add the result to the variance variable
    variance_pts += square_difference
# divided by the number and get the final variance
point_variance = variance_pts / len(nba_stats["pts"])
print(point_variance)


# Fractional Powers
a = 5 ** 2
# Raise to the fourth power
b = 10 ** 4

# Take the square root ( 3 * 3 == 9, so the answer is 3)
c = 9 ** (1/2)

# Take the cube root (4 * 4 * 4 == 64, so 4 is the cube root)
d = 64 ** (1/3)



# Calculating Standard Deviation
# A commonly used way to refer to how far data points are from the mean is called standard deviation.
"""
It is typical to measure what percentage of the data is within 1 standard deviation of the mean, or two standard deviations of the mean.
Standard deviation is a very useful concept, and is a great way to measure how spread out data is.
Luckily for us, standard deviation is just the square root of the variance.
"""
# Create a function to calculate for standard deviation
def standard_deviation(col):
    # Get the mean of the column first
    mean_col = nba_stats[col].mean()
    # set variance to 0
    variance = 0
    # for every index in the column
    for index in range(0,len(nba_stats[col])):
        # do the calculation
        square_difference = (nba_stats[col][index]-mean_col)**2
        variance += square_difference
    standard_deviation = (variance/len(nba_stats[col]))**(1/2)
    return standard_deviation

# call the function to perform standard deviation for the column
mp_dev = standard_deviation("mp")
ast_dev = standard_deviation("ast")
print(mp_dev)
print(ast_dev)



# Find Standard Deviation Distance
import matplotlib.pyplot as plt

plt.hist(nba_stats["pf"])
mean = nba_stats["pf"].mean()
plt.axvline(mean, color="r")
# We can calculate standard deviation by using the std() method on a pandas series.
std_dev = nba_stats["pf"].std()
# Plot a line one standard deviation below the mean
plt.axvline(mean - std_dev, color="g")
# Plot a line one standard deviation above the mean
plt.axvline(mean + std_dev, color="g")

# We can see how much of the data points fall within 1 standard deviation of the mean
# The more that falls into this range, the less spread out the data is
plt.show()

# We can calculate how many standard deviations a data point is from the mean by doing some subtraction and division
# First, we find the total distance for the first item by subtracting the mean
total_distance = nba_stats["pf"][0] - mean
# Then we divide by standard deviation to find how many standard deviations away the point is.
standard_deviation_distance = total_distance / std_dev

# Find how many standard deviations away from the mean point_10 is
point_10 = nba_stats["pf"][9] - mean
point_10_std = point_10 / std_dev
# Find how many standard deviations away from the mean point_100 is
point_100 = nba_stats["pf"][99] - mean
point_100_std = point_100 / std_dev
print(point_10_std)
print(point_100_std)



# Working With The Normal Distribution

import numpy as np
import matplotlib.pyplot as plt
# The norm module has a pdf function (pdf stands for probability density function)
from scipy.stats import norm

# The arange function generates a numpy vector
# The vector below will start at -1, and go up to, but not including 1
# It will proceed in "steps" of .01.  So the first element will be -1, the second -.99, the third -.98, all the way up to .99.
points = np.arange(-1, 1, 0.01)

# The norm.pdf function will take points vector and turn it into a probability vector
# Each element in the vector will correspond to the normal distribution (earlier elements and later element smaller, peak in the center)
# The distribution will be centered on 0, and will have a standard devation of .3
probabilities = norm.pdf(points, 0, .3)

# Plot the points values on the x axis and the corresponding probabilities on the y axis
# See the bell curve?
plt.plot(points, probabilities)
plt.show()


# Make a normal distribution across the range that starts at -10, ends at 10, and has the step .1.
# The distribution should have mean 0 and standard deviation of 2.
points2 = np.arange(-10, 10, 0.1)
probabilities2 = norm.pdf(points2, 0, 2)
plt.plot(points2, probabilities2)
plt.show()



# Normal Distribution Deviation
"""
One cool thing about normal distributions is that for every single one,
the same percentage of the data is within 1 standard deviation of the mean,
the same percentage is within 2 standard deviations of the mean, and so on.
About 68% of the data is within 1 standard deviation of the mean,
about 95% is within 2 standard deviations of the mean, and about 99% is within 3 standard deviations of the mean."""
# Housefly wing lengths in millimeters
wing_lengths = [36, 37, 38, 38, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 53, 53, 54, 55]
# For each point in wing_lengths, calculate the number of standard deviations away from the mean the point is.
mean_winglengths = sum(wing_lengths)/len(wing_lengths)
# list object has no attribute .std(), so std_dev_winglengths = wing_lengths.std() won't work
# list object has no attribute .mean() neither...
variance_first_step = [(i - mean_winglengths) ** 2 for i in wing_lengths]
variance_second_step = sum(variance_first_step)
variance_third_step = variance_second_step/len(wing_lengths)
std_dev_winglengths = variance_third_step ** (1/2)
dist_from_mean = []
for index in range(0, len(wing_lengths)):
    total_distance = wing_lengths[index] - mean_winglengths
    final = abs(total_distance)/std_dev_winglengths
    dist_from_mean.append(final)

# Compute what percentage of the data is within 1 standard deviation of the mean. Assign the result to within_one_percentage.
# Compute what percentage of the data is within 2 standard deviations of the mean. Assign the result to within_two_percentage
# Compute what percentage of the data is within 3 standard deviations of the mean. Assign the result to within_three_percentage.
within_one = 0
within_two = 0
within_three = 0
for index in range(0, len(dist_from_mean)):
    if dist_from_mean[index] <= 1:
        within_one += 1
    if dist_from_mean[index] <= 2:
        within_two += 1
    if dist_from_mean[index] <= 3:
        within_three += 1
within_one_percentage = within_one/len(dist_from_mean)
within_two_percentage = within_two/len(dist_from_mean)
within_three_percentage = within_three/len(dist_from_mean)

print(within_one_percentage)
print(within_two_percentage)
print(within_three_percentage)



# Plotting Correlations
# look at how two variables correlate with each other.
"""
A lot of statistics is about analyzing how variables impact each other, and the first step is to graph them out with a scatterplot.
While graphing them out, we can look at correlation.
If two variables both change together (ie, when one goes up, the other goes up), we know that they are correlated."""

import matplotlib.pyplot as plt

# This is plotting field goals attempted (number of shots someone takes in a season) vs point scored in a season
# Field goals attempted is on the x-axis, and points is on the y-axis
# As you can tell, they are very strongly correlated -- the plot is close to a straight line.
# The plot also slopes upward, which means that as field goal attempts go up, so do points.
# That means that the plot is positively correlated.
plt.scatter(nba_stats["fga"], nba_stats["pts"])
plt.show()

# If we make points negative (so the people who scored the most points now score the least, because 3000 becomes -3000), we can change the direction of the correlation
# Field goals are negatively correlated with our new "negative" points column -- the more free throws you attempt, the less negative points you score.
# We can see this because the correlation line slopes downward.
plt.scatter(nba_stats["fga"], -nba_stats["pts"])
plt.show()

# Now, we can plot total rebounds (number of times someone got the ball back for their team after someone shot) vs total assists (number of times someone helped another person score)
# These are uncorrelated, so you don't see the same nice line as you see with the plot above.
plt.scatter(nba_stats["trb"], nba_stats["ast"])
plt.show()

# Make a scatterplot of the "fta" (free throws attempted) column against the "pts" column
plt.scatter(nba_stats["fta"], nba_stats["pts"])
plt.show()

# Make a scatterplot of the "stl" (steals) column against the "pf" column.
plt.scatter(nba_stats["stl"], nba_stats["pf"])
plt.show()



# Measuring Correlation
# The most common way to measure correlation is to use Pearson's r, also called an r-value.
# An r-value ranges from -1 to 1, and indicates how strongly two variables are correlated.

from scipy.stats.stats import pearsonr

# The pearsonr function will find the correlation between two columns of data.
# It returns the r value and the p value.  We'll learn more about p values later on.
r, p_value = pearsonr(nba_stats["fga"], nba_stats["pts"])
# As we can see, this is a very high positive r value -- close to 1
print(r)

# These two columns are much less correlated
r, p_value = pearsonr(nba_stats["trb"], nba_stats["ast"])
# We get a much lower, but still positive, r value
print(r)

# Find the correlation between the "fta" field goals attempt column and the "pts" points column
r, p_value = pearsonr(nba_stats["fta"], nba_stats["pts"])
r_fta_pts = r
print(r_fta_pts)

# Find the correlation between the "stl" steal column and the "pf" foul column.
r, p_value = pearsonr(nba_stats["stl"], nba_stats["pf"])
r_stl_pf = r
print(r)



# Calculate Covariance
"""
Two variables are correlated when they both individually vary in similar ways.
For example, correlation occurs when if one variable goes up, another variable also goes up.
This is called covariance. Covariance is how things vary together."""
# The r-value is a ratio between the actual covariance, and the maximum possible positive covariance.
# The maximum possible covariance occurs when two variables vary perfectly (ie, you see a straight line on the plot).

# The nba_stats variable has been loaded.
# Make a function to compute covariance
def covariance(col1, col2):
    mean1 = sum(col1)/len(col1)
    mean2 = sum(col2)/len(col2)
    cov = 0
    for index in range(0,len(col1)):
        cov_step1 = (col1[index] - mean1)*(col2[index] - mean2)
        cov += cov_step1
    covariance = cov/len(col1)
    return covariance

# Use the function to compute the covariance of the "stl" and "pf" columns
cov_stl_pf = covariance(nba_stats["stl"], nba_stats["pf"])
# Use the function to compute the covariance of the "fta" and "pts" columns.
cov_fta_pts = covariance(nba_stats["fta"], nba_stats["pts"])
print(cov_stl_pf)
print(cov_fta_pts)




# Calculate Correlation coefficient
"""
calculate the correlation coefficient:
For the denominator, we need to multiple the standard deviations for x and y.
This is the maximum possible positive covariance -- it's just both the standard deviation values multiplied.
If we divide our actual covariance by this, we get the r-value."""
# You can use the std method on any Pandas Dataframe or Series to calculate the standard deviation. (Dataframe[col].std())
# You can use the cov function from NumPy to compute covariance, returning a 2x2 matrix. (cov(nba_stats["pf"], nba_stats["stl"])[0,1])
# Dataframe[col].var() can use to calculate variance
from numpy import cov
# The nba_stats variable has already been loaded.
# coefficient r = covariace/((std of A)*(std of B))
r_fta_blk = cov(nba_stats["fta"], nba_stats["blk"])[0,1]/(nba_stats["fta"].std() * nba_stats["blk"].std())
r_ast_stl = cov(nba_stats["ast"], nba_stats["stl"])[0,1]/(nba_stats["ast"].std() * nba_stats["stl"].std())
print(r_fta_blk)
print(r_ast_stl)








