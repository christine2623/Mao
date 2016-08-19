# Hypothesis Testing
"""
 hypothesis is a pattern or rule about a process in the world that can be tested.
 We use hypothesis testing to determine if a change we made had a meaningful impact or not.
"""
# Hypothesis testing allows us to calculate the probability that random chance was actually responsible for the difference in outcome.
"""
We first set up a null hypothesis that describes the status quo.
We then state an alternative hypothesis, which we used to compare with the null hypothesis to decide which describes the data better.
"""




# Research Design
"""
we invited 100 volunteers and split them into 2 even groups randomly:
Group A was given a placebo, or fake, pill and instructed to consume it on a daily basis.
Group B was given the actual weight loss pill and instructed to consume it on a daily basis.
Both groups were weighed before the study began and a month later, after the study ended."""





# Statistical Significance
"""
Statistics helps us determine if the difference in the weight lost between the 2 groups is
because of random chance or because of an actual difference in the outcomes.
If there is a meaningful difference, we say that the results are statistically significant.
"""
"""
Our null hypothesis should describe the default position of skepticism,
which is that there's no statistically significant difference between the outcomes of the 2 groups.
Put another way, it should state that any difference is due to random chance.
Our alternative hypothesis should state that there is in fact a statistically significant difference between the outcomes of the 2 groups.

Null hypothesis: participants who consumed the weight loss pills lost the same amount of weight as those who didn't take the pill.
Alternative hypothesis: participants who consumed the weight loss pills lost more weight than those who didn't take the pill.
"""

weight_lost_a = [3, 2, 3, 4, 3, 2, 2, 2, 1, 3, 2, 3, 1, 3, 4, 1, 3, 2, 1, 3, 4, 3, 2, 3, 7, 2, 3, 2, 5, 1, 1, 1, 3, 2, 4, 10, 2, 3, 2, 5, 6, 2, 3, 2, 3, 4, 1, 3, 3, 1]
weight_lost_b = [5, 4, 5, 5, 4, 5, 7, 5, 4, 3, 3, 5, 10, 3, 4, 9, 7, 6, 9, 4, 2, 5, 7, 7, 7, 5, 4, 8, 9, 6, 7, 6, 7, 6, 3, 5, 5, 4, 2, 3, 3, 5, 6, 9, 7, 6, 4, 5, 4, 3]

import numpy as np
import matplotlib.pyplot as plt

mean_group_a = np.mean(weight_lost_a)
mean_group_b = np.mean(weight_lost_b)
print(mean_group_a)
print(mean_group_b)

plt.hist(weight_lost_a)
plt.show()
plt.hist(weight_lost_b)
plt.show()





# Test Statistic
"""
To decide which hypothesis more accurately describes the data, we need to frame the hypotheses more quantitatively.
The first step is to decide on a test statistic, which is a numerical value that
summarizes the data and we can use in statistical formulas.
We use this test statistic to run a statistical test that will determine how likely the difference
between the groups were due to random chance.
"""
"""
Since we want to know if the amount of weight lost between the groups is meaningfully different,
we will use the difference in the means, also known as the mean difference,
of the amount of weight lost for each group as the test statistic.
"""
"""
Null hypothesis: x¯b−x¯a=0
Alternative hypothesis: x¯b−x¯a>0
"""
# Calculate the observed test statistic by subtracting mean_group_a from mean_group_b and assign to mean_difference.
mean_difference = np.mean(weight_lost_b) - np.mean(weight_lost_a)
print(mean_difference)





# Permutation Test
"""
The permutation test is a statistical test that involves
simulating rerunning the study many times and recalculating the test statistic for each iteration.
The goal is to calculate a distribution of the test statistics over these many iterations.
This distribution is called the sampling distribution and it approximates the full range of possible test statistics under the null hypothesis.
We can then benchmark the test statistic we observed in the data (a mean difference of 2.52) to determine
how likely it is to observe this mean difference under the null hypothesis.
If the null hypothesis is true, that the weight loss pill doesn't help people lose more weight,
than the observed mean difference of 2.52 should be quite common in the sampling distribution.
If it's instead extremely rare, then we accept the alternative hypothesis instead.
"""
"""
To simulate rerunning the study, we randomly reassign each data point (weight lost) to either group A or group B.
We keep track of the recalculated test statistics as a separate list.
"""
"""
Ideally, the number of times we re-randomize the groups that each data point belongs to matches the total number of possible permutations.
Usually, the number of total permutations is too high for even powerful supercomputers to calculate within a reasonable time frame.
While we'll use 1000 iterations for now since we'll get the results back quickly,
in later missions we'll learn how to quantify the tradeoff we make between accuracy and speed to determine the optimal number of iterations.
"""
all_values = [3, 5, 2, 4, 3, 5, 4, 5, 3, 4, 2, 5, 2, 7, 2, 5, 1, 4, 3, 3, 2, 3, 3, 5, 1, 10, 3, 3, 4, 4, 1, 9, 3, 7, 2, 6, 1, 9, 3, 4, 4, 2, 3, 5, 2, 7, 3, 7, 7, 7, 2, 5, 3, 4, 2, 8, 5, 9, 1, 6, 1, 7, 1, 6, 3, 7, 2, 6, 4, 3, 10, 5, 2, 5, 3, 4, 2, 2, 5, 3, 6, 3, 2, 5, 3, 6, 2, 9, 3, 7, 4, 6, 1, 4, 3, 5, 3, 4, 1, 3]
# Create an empty list named mean_differences
mean_differences = []
# Inside a for loop that repeats 1000 times:
for times in range(1000):
    # Assign empty lists to the variables group_a and group_b
    group_a = []
    group_b = []
    # Inside a for loop that iterates over all_values
    for index in range(0, len(all_values)):
        # Use the numpy.random.rand() function to generate a value between 0 and 1.
        # numpy.random.rand(d0, d1, ..., dn) -> Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
        random = np.random.rand()
        # If the random value is larger than or equal to 0.5, assign that weight loss value to group A
        if random >= 0.5:
            group_a.append(all_values[index])
        # If the random value is less than 0.5, assign that weight loss value to group B
        else:
            group_b.append(all_values[index])
    # Use the numpy.mean() function to calculate the means of group_a and group_b.
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    # Subtract the mean of group A from group B
    iteration_mean_difference = mean_b - mean_a
    # Append iteration_mean_difference to mean_differences
    mean_differences.append(iteration_mean_difference)
# Use plt.hist() to generate a histogram of mean_differences.
plt.hist(mean_differences)
plt.show()





# Sampling Distribution
"""
create a dictionary that contains the values in the sampling distribution so we can benchmark our observed test statistic against it.
The keys in the dictionary should be the test statistic and the values should be their frequency."""




# Dictionary Representation Of A Distribution
"""
To check if a key exists in a dictionary, we need to use the get() method to:
return the value at the specified key if it exists in the dictionary or
return another value we specify instead.

Here are the parameters the method takes in:
the required parameter is the key we want to look up,
the optional parameter is the default value we want returned if the key is not found."""
""" Example:
empty = {}

# Since "a" isn't a key in empty, the value False is returned.
key_a = empty.get("a", False):

empty["b"] = "boat"

# key_b is the value for the key "b" in empty.
key_b = empty.get("b", False):
# "boat" is assigned to key_b.
"""
""" Example 2:
empty = {"c": 1}
if empty.get("c", False):
    # If in the dictionary, grab the value, increment by 1, reassign.
    val = empty.get("c")
    inc = val + 1
    empty["c"] = inc
else:
    # If not in the dictionary, assign `1` as the value to that key.
    empty["c"] = 1
"""
# Create an empty dictionary called sampling_distribution
# whose keys will be the test statistics and whose values are the frequencies of the test statistics.
sampling_distribution = {}
# Inside a for loop that iterates over mean_differences, check if each value exists as a key in sampling_distribution:
for index in range(0, len(mean_differences)):
# Use the dictionary method get() with a default condition of False to check
    # if the current iteration's value is already in sampling_distribution.
    if sampling_distribution.get(mean_differences[index], False):
        # If it is, increment the existing value in sampling_distribution for that key by 1.
        # count is the value of the key in the dictionary
        count = sampling_distribution.get(mean_differences[index])
        # increase the count by 1
        inc = count + 1
        # assign the new value back to the key in the dictionary
        sampling_distribution[mean_differences[index]] = inc
        # another way to perform the above task: sampling_distribution[mean_differences[index]] = sampling_distribution[mean_differences[index]] + 1
    # If it isn't, add it to sampling_distribution as a key and assign 1 as the value.
    else:
        # assign the value of the key in the dictionary to 1
        sampling_distribution[mean_differences[index]] = 1





# P Value
"""
In the sampling distribution we generated, most of the values are closely centered around the mean difference of 0.
This means that if it were purely up to chance, both groups would have lost the same amount of weight (the null hypothesis).
But since the observed test statistic is not near 0,
it could mean that the weight loss pills could be responsible for the mean difference in the study."""
"""
We can now use the sampling distribution to determine the number of times a value of 2.52 or higher appeared in our simulations.
If we then divide that frequency by 1000, we'll have the probability of observing a mean difference of 2.52 or higher purely due to random chance.
This probability is called the p value.
If this value is high, it means that the difference in the amount of weight both groups lost could have easily happened randomly
and the weight loss pills probably didn't play a role.
On the other hand, a low p value implies that there's an incredibly small probability that the mean difference we observed
was because of random chance.
"""
"""
if the p value is less than the threshold, we:
reject the null hypothesis that there's no difference in mean amount of weight lost by participants in both groups,
accept the alternative hypothesis that the people who consumed the weight loss pill lost more weight,
conclude that the weight loss pill does affect the amount of weight people lost.

if the p value is greater than the threshold, we:
accept the null hypothesis that there's no difference in the mean amount of weight lost by participants in both groups,
reject the alternative hypothesis that the people who consumed the weight loss pill lost more weight,
conclude that the weight loss pill doesn't seem to be effective in helping people lose more weight.

The most common p value threshold is 0.05 or 5%, which is what we'll use in this mission.
Although .05 is an arbitrary threshold, it means that there's only a 5% chance that the results are due to random chance,
which most researchers are comfortable with."""
# Create an empty list named frequencies
frequencies = []
# Inside a for loop that iterates over the keys in sampling_distribution
for keys in sampling_distribution:
    # If the key is 2.52 or larger, add its value to frequencies
    if keys >= 2.52:
        frequencies.append(sampling_distribution[keys])
    else:
        continue
# use the NumPy function sum() to calculate the sum of the values in frequencies.
total_freq = np.sum(frequencies)
# Divide the sum by 1000 and assign to p_value.
p_value = total_freq / 1000

print(p_value)

"""Findings:
Since the p value of 0 is less than the threshold we set of 0.05,
we conclude that the difference in weight lost can't be attributed to random chance alone.
We therefore reject the null hypothesis and accept the alternative hypothesis.
"""





# Caveats
"""
A few caveats:
Research design is incredibly important and can bias your results.
For example, if the participants in group A realized they were given placebo sugar pills,
they may modify their behavior and affect the outcome.
The p value threshold you set can also affect the conclusion you reach.
If you set too high of a p value threshold, you may accept the alternative hypothesis incorrectly and
fail to reject the null hypothesis. This is known as a type I error.
If you set too low of a p value threshold, you may reject the alternative hypothesis incorrectly
in favor of accepting the null hypothesis. This is known as a type II error.
"""





