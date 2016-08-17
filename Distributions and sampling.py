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
mid_sample = probability_of_one(300, 1000)
plt.hist(mid_sample, 20)
plt.show()


