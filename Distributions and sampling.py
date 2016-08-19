import pandas as pd

income = pd.read_csv("income.csv")

# The first 5 rows of the data.
print(income.head())

# Find the county with the lowest median income in the US (median_income).
# DataFrame.idxmin(axis=0, skipna=True)
# Return index of first occurrence of minimum over requested axis. NA/null values are excluded
lowest_income_county = income["county"][income["median_income"].idxmin()]

# Find the county that has more than 50000 residents with the lowest median income.
high_pop = income[income["pop_over_25"]>50000]
# The column you want put in the front, condition put after it
lowest_income_high_pop_county = income["county"][high_pop["median_income"].idxmin()]





# Random Numbers
"""
Sometimes, instead of looking at a whole dataset, you just want to take a sample of it.
This usually happens when dealing with the whole set of data is impractical."""
# The first step is to generate random numbers. We can use the random package in Python to do this for us.
import random

# Returns a random integer between the numbers 0 and 10, inclusive.
num = random.randint(0, 10)

# Generate a sequence of 10 random numbers between the values of 0 and 10.
random_sequence = [random.randint(0, 10) for _ in range(10)]

# Sometimes, when we generate a random sequence, we want it to be the same sequence whenever the program is run.
# An example is when you use random numbers to select a subset of the data, and you want other people
# looking at the same data to get the same subset.
# We can ensure this by setting a random seed.
# A random seed is an integer that is used to "seed" a random number generator.
# After a random seed is set, the numbers generated after will follow the same sequence.
random.seed(10)
print([random.randint(0,10) for _ in range(5)])
random.seed(10)
# Same sequence as above.
print([random.randint(0,10) for _ in range(5)])
random.seed(11)
# Different seed means different sequence.
print([random.randint(0,10) for _ in range(5)])

# Set a random seed of 20 and generate a list of 10 random numbers between the values 0 and 10.
random.seed(20)
new_sequence = [random.randint(0,10) for _ in range(10)]
print(new_sequence)




# Selecting Items From A List
# The easiest way is to use the random.sample method to select a specified number of items from a list.
# Let's say that we have some data on how much shoppers spend in a store.
shopping = [300, 200, 100, 600, 20]

# We want to sample the data, and only select 4 elements.

random.seed(1)
shopping_sample = random.sample(shopping, 4)

# 4 random items from the shopping list.
print(shopping_sample)





# Population Vs Sample
import matplotlib.pyplot as plt

# A function that returns the result of a die roll.
def roll():
    return random.randint(1, 6)

random.seed(1)
small_sample = [roll() for _ in range(10)]

# Plot a histogram with 6 bins (1 for each possible outcome of the die roll)
plt.hist(small_sample, 6)
plt.show()

# Set the random seed to 1, then generate a medium sample of 100 die rolls. Plot the result using a histogram with 6 bins.
random.seed(1)
mid_roll = [roll() for _ in range(100)]
plt.hist(mid_roll, 6)
plt.show()

# Set the random seed to 1, then generate a large sample of 10000 die rolls. Plot the result using a histogram with 6 bins.
random.seed(1)
large_roll = [roll() for _ in range(10000)]
plt.hist(large_roll, 6)
plt.show()





# Finding The Right Sample Size
def probability_of_one(num_trials, num_rolls):
    """
    This function will take in the number of trials, and the number of rolls per trial.
    Then it will conduct each trial, and record the probability of rolling a one.
    """
    probabilities = []
    for i in range(num_trials):
        die_rolls = [roll() for _ in range(num_rolls)]
        one_prob = len([d for d in die_rolls if d==1]) / num_rolls
        probabilities.append(one_prob)
    return probabilities

random.seed(1)
small_sample = probability_of_one(300, 50)
plt.hist(small_sample, 20)
plt.show()

# Set the random seed to 1, then generate probabilities for 300 trials of 100 die rolls each. Make a histogram with 20 bins.
random.seed(1)
mid_sample = probability_of_one(300, 100)
plt.hist(mid_sample, 20)
plt.show()

# Set the random seed to 1, then generate probabilities for 300 trials of 1000 die rolls each. Make a histogram with 20 bins.
random.seed(1)
large_sample = probability_of_one(300, 1000)
plt.hist(large_sample, 20)
plt.show()




# What Are The Odds?
"""
if we do 100 rolls of the die, and get a .25 probability of rolling a 1,
we could look up how many trials in our data above got that probability or higher for one."""

import numpy
# Use numpy.std(list object) to get the std! numpy.mean(list) to get the mean.
large_sample_std = numpy.std(large_sample)
large_sample_mean = numpy.mean(large_sample)

# Find how many standard deviations away from the mean of large_sample .18 is.
Difference = 0.18 - large_sample_mean
deviations_from_mean = Difference / large_sample_std
print(deviations_from_mean)

# Find how many probabilities in large sample are greater than or equal to .18.
greater_than_18 = [i for i in large_sample if i >= 0.18]
over_18_count = len(greater_than_18) / len(large_sample)
print(over_18_count)





# Sampling Counties
# This is the mean median income in any US county.
# Dataframe[col].mean() ; Series.mean()
mean_median_income = income["median_income"].mean()
print(mean_median_income)

def get_sample_mean(start, end):
    return income["median_income"][start:end].mean()

def find_mean_incomes(row_step):
    mean_median_sample_incomes = []
    # Iterate over the indices of the income rows
    # Starting at 0, and counting in blocks of row_step (0, row_step, row_step * 2, etc).
    for i in range(0, income.shape[0], row_step):
        # Find the mean median for the row_step counties from i to i+row_step.
        mean_median_sample_incomes.append(get_sample_mean(i, i+row_step))
    return mean_median_sample_incomes

nonrandom_sample = find_mean_incomes(100)
plt.hist(nonrandom_sample, 20)
plt.show()

# What you're seeing above is the result of biased sampling.
# Instead of selecting randomly, we selected counties that were next to each other in the data.
# This picked counties in the same state more often that not, and created means that didn't represent the whole country.
# This is the danger of not using random sampling -- you end up with samples that don't reflect the entire population.
# This gives you a distribution that isn't normal.

import random
def select_random_sample(count):
    # From index 0 to income.shape[0], random pick "count" number of indices
    random_indices = random.sample(range(0, income.shape[0]), count)
    return income.iloc[random_indices]

random.seed(1)

# Use the select_random_sample function to pick 1000 random samples of 100 counties each from the income data.
# Find the mean of the median_income column for each sample.
random_select_result = [select_random_sample(100)["median_income"].mean() for _ in range(1000)]

# Plot a histogram with 20 bins of all the mean median incomes.
plt.hist(random_select_result, 20)
plt.show()



# An Experiment
"""
We want to run an experiment to see whether a certain kind of adult education can help high school graduates
earn more relative to college graduates than they could otherwise.
We decide to trial our program in 100 counties, and measure the median incomes of both groups in 5 years.
At the end of 5 years, we first need to measure the whole population to determine the typical ratio
between high school graduate earnings and college graduate earnings."""

def select_random_sample(count):
    random_indices = random.sample(range(0, income.shape[0]), count)
    return income.iloc[random_indices]

random.seed(1)
# Select 1000 random samples of 100 counties each from the income data using the select_random_sample method.
# For each sample:  Divide the median_income_hs column by median_income_college to get ratios.
# Then, find the mean of all the ratios in the sample.
# Add it to the list, mean_ratios.
# mean_ratios = [select_random_sample(100)(["median_income_hs"]/["median_income_college"]).mean() for _ in range(1000)] doesn't work
# list.mean() won't work!! Only series can use .mean()
mean_ratios = []
for i in range(1000):
    sample = select_random_sample(100)
    sample_ratio = sample["median_income_hs"]/sample["median_income_college"]
    mean_ratios.append(sample_ratio.mean())
# Plot a histogram containing 20 bins of the mean_ratios list.
plt.hist(mean_ratios, 20)
plt.show()

""" Findings:
After 5 years, we determine that the mean ratio in our random sample of 100 counties is .675 --
that is, high school graduates on average earn 67.5% of what college graduates do."""



# Statistical Significance
"""
Statistical significance is used to determine if a result is valid for a population or not.
You usually set a significance level beforehand that will determine if your hypothesis is true or not.
After conducting the experiment, you check against the significance level to determine.
A common significance level is .05. This means: "only 5% or less of the time will the result have been due to chance"."""

significance_value = None

# Determine how many values in mean_ratios are greater than or equal to .675.
greater_count = [i for i in mean_ratios if i >= .675]
# Divide by the total number of items in mean_ratios to get the significance level.
significance_value = len(greater_count)/len(mean_ratios)
print(significance_value)

"""Findings:
Our significance value was .014. Based on the entire population,
only 1.4% of the time will the wage results we saw have occurred on their own.
So our experiment exceeded our significance level (lower means more significant).
Thus, our experiment showed that the program did improve the wages of high school graduates relative to college graduates."""




# Final Result
"""
You need a larger deviation from the mean to have something be "significant" if your sample size is smaller.
The larger the trial, the smaller the deviation needs to be to get a significant result."""
# This is "steeper" than the graph from before, because it has 500 items in each sample.
random.seed(1)
mean_ratios = []
for i in range(1000):
    sample = select_random_sample(500)
    ratios = sample["median_income_hs"] / sample["median_income_college"]
    mean_ratios.append(ratios.mean())

plt.hist(mean_ratios, 20)
plt.show()