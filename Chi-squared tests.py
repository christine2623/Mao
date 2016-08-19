# Observed And Expected Frequencies
# chi-square tests enables us to determine the statistical significance of observing a set of categorical values.
# The chi-squared test enables us to quantify the difference between sets of observed and expected categorical values.




# Calculating Differences
# One way that we can determine the differences between observed and expected values is to compute simple proportional differences.
# our observed values were 10771 Females, and 21790 Males. Our expected values were 16280.5 Females and 16280.5 Males.
# Compute the proportional difference in number of observed Females vs number of expected Females.
female_diff = (10771 - 16280.5)/16280.5
# Compute the proportional difference in number of observed Males vs number of expected Males.
male_diff = (21790 - 16280.5)/16280.5
print(female_diff)
print(male_diff)




# Updating The Formula
"""
What we really want to find is one number that can tell us how much all of our observed counts deviate
from all of their expected counterparts. This will let us figure out if our difference in counts is statistically significant.
We can get one step closer to this by squaring the top term in our difference formula.

Squaring the difference will ensure that all the differences don't sum to zero (you can't have negative squares),
giving us a non-zero number we can use to assess statistical significance.

We can calculate χ**2, the chi-squared value, by adding up all of the squared differences between observed and expected values."""
# Compute the difference in number of observed Females vs number of expected Females using the updated technique.
female_diff = ((10771 - 16280.5)**2)/16280.5
# Compute the difference in number of observed Males vs number of expected Males using the updated technique.
male_diff = ((21790 - 16280.5)**2)/16280.5

gender_chisq = male_diff + female_diff
print(gender_chisq)




# Generating A Distribution
"""
Now that we have a chi-squared value for our observed and expected gender counts,
we need a way to figure out what the chi-squared value represents.
We can translate a chi-squared value into a statistical significance value using a chi-squared sampling distribution.

 A p-value allows us to determine whether the difference between two values is due to chance, or due to an underlying difference."""
"""
We can generate a chi-squared sampling distribution using our expected probabilities.
If we repeatedly generate random samples that contain 32561 samples, and graph the chi-squared value of each sample,
we'll be able to generate a distribution. Here's a rough algorithm:

Randomly generate 32561 numbers that range from 0-1.
Based on the expected probabilities, assign Male or Female to each number.
Compute the observed frequences of Male and Female.
Compute the chi-squared value and save it.
Repeat several times.
Create a histogram of all the chi-squared values.

By comparing our chi-squared value to the distribution, and seeing what percentage of the distribution is greater than our value,
we'll get a p-value. For instance, if 5% of the values in the distribution are greater than our chi-squared value, the p-value is .05."""
import numpy as np
import matplotlib.pyplot as plt
chi_squared_values = []
# Inside a for loop that repeats 1000 times
for time in range(1000):
    # Use the numpy.random.random function to generate 32561 numbers between 0.0 and 1.0
    # numpy.random.random() : Return random floats in the half-open interval [0.0, 1.0).
    # pass in 32561, into the function to get a vector with 32561 elements.
    random = np.random.random(32561,)
    # For each of the numbers, if it is less than .5, replace it with 0, otherwise replace it with 1
    for index in range(0, len(random)):
        if random[index] < 0.5:
            random[index] = 0
        # another way -> random[index][random[index] < .5] = 0
        else:
            random[index] = 1
    # Count up how many times 0 occurs (Male frequency), and how many times 1 occurs (Female frequency)
    male_count = 0
    female_count = 0
    for index in range(0, len(random)):
        if random[index] == 0:
            male_count += 1
        # another way -> male_count = len(sequence[sequence == 0])
        else:
            female_count += 1
    # Compute male_diff by subtracting the expected Male count from the observed Male count, squaring it, and dividing by the expected Male count.
    male_diff = ((male_count - 16280.5)**2)/1680.5
    # Compute female_diff by subtracting the expected Female count from the observed Female count, squaring it, and dividing by the expected Female count.
    female_diff = ((female_count - 16280.5) ** 2) / 1680.5
    # Add up male_diff and female_diff to get the chi-squared vlaue.
    chi_values = male_diff + female_diff
    # Append the chi-squared value to chi_squared_values.
    chi_squared_values.append(chi_values)

plt.hist(chi_squared_values)
plt.show()

""" Findings:
In the last screen, our calculated chi-squared value is greater than all the values in the distribution, so our p-value is 0,
indicating that our result is statistically significant.
You may recall from the last mission that .05 is the typical threshold for statistical significance,
and anything below it is considered significant."""




# Statistical Significance
""" we get a p-value of 0. This means that there is a 0% chance that we could get such a result randomly.
Indicate that we need to investigate our data collection techniques more closely to figure out why such a result occurred."""




# Smaller Samples
# One interesting thing about chi-squared values is that they get smaller as the sample size decreases.
# As sample size changes, the chi-squared value changes proportionally.
# Let's say our observed values are 107.71 Females, and 217.90 Males. Our expected values are 162.805 Females and 162.805 Males.
# Compute the difference in number of observed Females vs number of expected Females using the new formula.
female_diff = ((107.71 - 162.805)**2)/162.805
# Compute the difference in number of observed Males vs number of expected Males using the new formula.
male_diff = ((217.90 - 162.805)**2)/162.805
# Add male_diff and female_diff together and assign to the variable gender_chisq
gender_chisq = female_diff + male_diff
print(gender_chisq)





# Sampling Distribution Equality
# As sample sizes get larger, seeing large deviations from the expected probabilities gets less and less likely.
"""Chi-squared values for the same sized effect increase as sample size increases,
but the chance of getting a high chi-squared value decreases as the sample gets larger.
These two effects offset each other, and a chi-squared sampling distribution constructed when sampling 200 items for each iteration
will look identical to one sampling 1000 items.
This enables us to easily compare any chi-squared value to a master sampling distribution
to determine statistical significance, no matter what sample size the chi-squared value was created with."""
chi_squared_values = []
for time in range(1000):
    # Use the numpy.random.random function to generate 300 numbers between 0.0 and 1.0
    # numpy.random.random() : Return random floats in the half-open interval [0.0, 1.0).
    # pass in 300, into the function to get a vector with 300 elements.
    random = np.random.random(300,)
    # For each of the numbers, if it is less than .5, replace it with 0, otherwise replace it with 1
    for index in range(0, len(random)):
        if random[index] < 0.5:
            random[index] = 0
        # another way -> random[index][random[index] < .5] = 0
        else:
            random[index] = 1
    # Count up how many times 0 occurs (Male frequency), and how many times 1 occurs (Female frequency)
    male_count = 0
    female_count = 0
    for index in range(0, len(random)):
        if random[index] == 0:
            male_count += 1
        # another way -> male_count = len(sequence[sequence == 0])
        else:
            female_count += 1
    # Compute male_diff by subtracting the expected Male count from the observed Male count, squaring it, and dividing by the expected Male count.
    male_diff = ((male_count - 150)**2)/150
    # Compute female_diff by subtracting the expected Female count from the observed Female count, squaring it, and dividing by the expected Female count.
    female_diff = ((female_count - 150) ** 2) / 150
    # Add up male_diff and female_diff to get the chi-squared vlaue.
    chi_values = male_diff + female_diff
    # Append the chi-squared value to chi_squared_values.
    chi_squared_values.append(chi_values)

plt.hist(chi_squared_values)
plt.show()





# Degrees Of Freedom
"""
A degree of freedom is the number of values that can vary without the other values being "locked in".
In the case of our two categories, there is actually only one degree of freedom.
Degrees of freedom are an important statistical concept that will come up repeatedly, both in this mission and after.
"""




# Increasing Degrees Of Freedom
"""
We can actually work with any number of categories, and any number of degrees of freedom.
We can accomplish this using largely the same formula we've been using,
but we will need to generate new sampling distributions for each number of degrees of freedom."""
# If we look at the race column of the income data, the possible values are White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, and Other.
diffs = []
observed = [27816, 3124, 1039, 311, 271]
expected = [26146.5, 3939.9, 944.3, 260.5, 1269.8]

# for i,j in enumerate(list) will print out the values along with the index
for i, obs in enumerate(observed):
    exp = expected[i]
    diff = (obs - exp) ** 2 / exp
    diffs.append(diff)

race_chisq = sum(diffs)





# Using SciPy
"""Rather than constructing another chi-squared sampling distribution for 4 degrees of freedom,
we can use a function from the SciPy library to do it more quickly.
The scipy.stats.chisquare function takes in an array of observed frequences, and an array of expected frequencies,
and returns a tuple containing both the chi-squared value and the matching p-value that we can use to check for statistical significance.
"""
"""Example
import numpy as np
from scipy.stats import chisquare

observed = np.array([5, 10, 15])
expected = np.array([7, 11, 12])
chisquare_value, pvalue = chisquare(observed, expected)

The scipy.stats.chisquare function returns a list,
so we can assign each item in the list to a separate variable using 2 variable names separated with a comma."""
from scipy.stats import chisquare
import numpy as np
observed = np.array([27816, 3124, 1039, 311, 271])
expected = np.array([26146.5, 3939.9, 944.3, 260.5, 1269.8])
chisq_value, race_pvalue = chisquare(observed, expected)
print(chisq_value)
print(race_pvalue)