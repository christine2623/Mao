# Binomial Distributions
"""
In the last mission, we defined pp as the probability of an outcome occurring, and qq as the probability of it not occuring, where q=1−pq=1−p.
These types of probabilites are known as binomial -- there are two values, which add to 1 collectively.
There's a 100% chance of one outcome or the other occurring.

These can range from testing whether changing the button color on your webpage increases conversion rate to seeing if a new drug increases patient recovery rate.
"""




# Bikesharing Distribution
import pandas
bikes = pandas.read_csv("bike_rental_day.csv")

# find the probability of there being more than 5000 riders in a single day.
over_5000 = bikes[bikes["cnt"] > 5000].shape[0]
prob_over_5000 = over_5000/bikes.shape[0]
print(prob_over_5000)



# Computing The Distribution
# Using the knowledge from the last mission, create a function that can compute the probability of k outcomes out of N events occuring.
import math
def combinations(total_num, num):
    numerator = math.factorial(total_num)
    denominator = math.factorial(num) * math.factorial(total_num - num)
    return numerator/denominator

def prob_single_event(p, q, total_num, num):
    return (p ** num) * (q ** (total_num - num))

def prob_combination(p, q, total_num, num):
    return combinations(total_num, num) * prob_single_event(p, q, total_num, num)

# An outcome is a day where there are more than 5000 riders, with p=.39
# You should have a list with 31 items, where the first item is the probability of 0 days out of 30 with more than 5000 riders, the second is the probability of 1 day out of 30, and so on, up to 30 days out of 30.
# Assign the list to outcome_probs.
# list() make it list data type
outcome_counts = list(range(31))
outcome_probs = []
for i in outcome_counts:
    result = prob_combination(0.39, 0.61, 30, i)
    outcome_probs.append(result)
print(outcome_probs)





# Plotting The Distribution
# The points in our data are discrete and not continuous, so we use a bar chart when plotting.
import matplotlib.pyplot as plt
# The most likely number of days is between 10 and 15.
plt.bar(outcome_counts, outcome_probs)
plt.show()





# Simplifying The Computation
# We can instead use the binom.pmf function from SciPy to do this faster.
"""Example:
from scipy import linspace
from scipy.stats import binom

# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
outcome_counts = linspace(0,30,31)

# Create the binomial probabilities, one for each entry in outcome_counts.
dist = binom.pmf(outcome_counts,30,0.39)"""
# The pmf function in SciPy is an implementation of the mathematical probability mass function.
# The pmf will give us the probability of each k in our outcome_counts list occurring.
"""
A binomial distribution only needs two parameters.
A parameter is the statistical term for a number that summarizes data for the entire population.

For a binomial distribution, the parameters are:
N, the total number of events,
p, the probability of the outcome we're interested in seeing.

The SciPy function pmf matches this and takes in the following parameters:
x: the list of outcomes,
n: the total number of events,
p: the probability of the outcome we're interested in seeing.
Because we only need two parameters to describe a distribution,
it doesn't matter whether we want to know if it will be sunny 5 days out of 5,
or if 5 out of 5 coin flips will turn up heads.
As long as the outcome that we care about has the same probability (p), and N is the same,
the binomial distribution will look the same."""

import scipy
from scipy import linspace
from scipy.stats import binom

# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry). Outcome list
outcome_counts = linspace(0,30,31)

# Generate a binomial distribution, and then find the probabilities for each value in outcome_counts.
# Use N=30, and p=.39
dist = binom.pmf(outcome_counts,30,0.39)

# Plot the resulting data as a bar chart.
# plt.bar(location of bars, what to be shown)
plt.bar(outcome_counts, dist)
plt.show()




# How To Think About A Probability Distribution
"""
A probability distribution is a great way to visualize data, but bear in mind that it's not dealing in absolute values.
A probability distribution can only tell us which values are likely, and how likely they are."""




# Computing The Mean Of A Probability Distribution
"""
Sometimes we'll want to be able to tell people the expected value of a probability distribution --
the most likely result of a single sample that we look at.
To compute this, we just multiply N by p."""

dist_mean = None
# Compute the mean for the bikesharing data, where N=30, and p=.39.
N = 30
p = .39
dist_mean = N * p
print(dist_mean)





# Computing The Standard Deviation
# This helps us find how much the actual values will vary from the mean when we take a sample.
# The formula for standard deviation of a probability distribution is: (N*p*q)**(1/2)
dist_stdev = None
# Compute the standard deviation for the bikesharing data, where N=30N=30, and p=.39p=.39.
dist_stdev = (N*p*(1-p))**(1/2)
""" Another way to perform this:
import math
dist_stdev = math.sqrt(30 * .39 * .61)"""
print(dist_stdev)






# A Different Plot
# Generate a binomial distribution, with N=10, and p=.39.
# Find the probabilities for each value in outcome_counts.
# Plot the resulting data as a bar chart.
outcome_counts = linspace(0,10,11)
dist1 = binom.pmf(outcome_counts,10,0.39)
plt.bar(outcome_counts, dist1)
plt.show()

# Generate a binomial distribution, with N=100, and p=.39.
# Find the probabilities for each value in outcome_counts.
# Plot the resulting data as a bar chart.
outcome_counts = linspace(0,100,101)
dist2 = binom.pmf(outcome_counts,100,0.39)
plt.bar(outcome_counts, dist2)
plt.show()





# The Normal Distribution
# the more events we looked at, the closer our distribution was to being normal.
# Create a range of numbers from 0 to 100, with 101 elements (each number has one entry).
outcome_counts = scipy.linspace(0,100,101)

# Create a probability mass function along the outcome_counts.
outcome_probs = binom.pmf(outcome_counts,100,0.39)

# Plot a line, not a bar chart.
# plt.plot(location of bars, what to be shown)
plt.plot(outcome_counts, outcome_probs)
plt.show()





# Cumulative Density Function
""" Example:
from scipy import linspace
from scipy.stats import binom

# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
outcome_counts = linspace(0,30,31)

# Create the cumulative binomial probabilities, one for each entry in outcome_counts.
dist = binom.cdf(outcome_counts,30,0.39)"""

# Create a cumulative distribution where N=30N=30 and p=.39p=.39 and generate a line plot of the distribution.
outcome_counts = linspace(0,30,31)
dist = binom.cdf(outcome_counts, 30, 0.39)
plt.plot(outcome_counts, dist)
plt.show()





# Calculating Z-Scores
"""
We can calculate z-scores (the number of standard deviations away from the mean a probability is) fairly easily.
These z-scores can then be used how we used z-scores earlier --
to find the percentage of values to the left and right of the value we're looking at.

To make this more concrete, say we had 16 days where we observed more than 5000 riders.
Is this likely? Unlikely? Using a z-score, we can figure out exactly how common this event is.

This is because every normal distribution, as we learned in an earlier mission,
has the same properties when it comes to what percentage of the data is within a certain number of standard deviations of the mean.
About 68% of the data is within 1 standard deviation of the mean, 95% is within 2, and 99% is within 3."""
# If we want to figure out the number of standard deviations from the mean a value is, we just do: (k-mean)/standard_dev
# k is the number of event occurred
"""
Based on the standard z-score table, this is unlikely -- a 2.8% chance.
This tells us that 97.2% of the data is within 2.2 standard deviations of the mean,
so a result to be as different from the mean as this, there is a 2.8% probability that it occurred by chance.

Note that this means both "sides" of the distribution.
There's a 1.4% chance that a value is 2.2 standard deviations or more above the mean (to the right of the mean),
and there's a 1.4% chance that a value is 2.2 standard deviation below the mean (to the left)."""






# Faster Way To Calculate Likelihood
"""
We don't want to have to use a z-score table every time we want to see how likely or unlikely a probability is.
A much faster way is to use the cumulative distribution function (cdf) like we did earlier.
This won't give us the exact same values as using z-scores, because the distribution isn't exactly normal,
but it will give us the actual amount of probability in a distribution to the left of a given k."""
""" Example:
# The sum of all the probabilities to the left of k, including k.
left = binom.cdf(k,N,p)

# The sum of all probabilities to the right of k.
right = 1 - left"""
left_16 = None
right_16 = None
# Find the probability to the left of k=16 (including 16) when N=30 and p=.39.
left_16 = binom.cdf(16, 30, .39)
# Find the probability to the right of k=16 when N=30 and p=.39.
right_16 = 1 - left_16

print(left_16)
print(right_16)








