# Probability Of Renting Bikes
import pandas
bikes = pandas.read_csv("bike_rental_day.csv")

# Find the number of days the bikes rented exceeded the threshold.
days_over_threshold = bikes[bikes["cnt"] > 2000].shape[0]
# Find the total number of days we have data for.
total_days = bikes.shape[0]

# Get the probability that more than 2000 bikes were rented for any given day.
probability_over_2000 = days_over_threshold / total_days
print(probability_over_2000)

# Find the probability that more than 4000 bikes were rented on any given day
over_4000 = bikes[bikes["cnt"] > 4000].shape[0]
probability_over_4000 = over_4000 / total_days
print(probability_over_4000)





# Calculating Probabilities
# calculate the probability that 1 coin out of 3 is heads
coin_1_prob = .5 ** 3 + .5 ** 3 + .5 ** 3
print(coin_1_prob)




# Calculating The Number Of Combinations
"""
Let's say that we live in Los Angeles, CA, and the chance of any single day being sunny is .7.
The chance of a day not being sunny is .3."""
# Find the number of combinations in which 1 day will be sunny.
sunny_1_combinations = None
sunny_1_combinations = 5
print(sunny_1_combinations)




# Number Of Combinations Formula
"""
We can calculate the number of combinations in which an outcome can occur k times in a set of events with a formula:
N!/(k!(N−k)!)
In this formula, NN is the total number of events we have,
and kk is the target number of times we want our desired outcome to occur.
The !! symbol means factorial. A factorial means "multiply every number from 1 to this number together"."""




# Finding The Number Of Combinations
# find the probability that in 10 days, 7 or more days have more than 4000 riders
import math
def find_outcome_combinations(N, k):
    # Calculate the numerator of our formula.
    numerator = math.factorial(N)
    # Calculate the denominator.
    denominator = math.factorial(k) * math.factorial(N - k)
    # Divide them to get the final value.
    return numerator / denominator

combinations_7 = find_outcome_combinations(10, 7)

# Find the number of combinations where 8 days out of 10 have more than 4000 rentals
combinations_8 = find_outcome_combinations(10, 8)

# Find the number of combinations where 9 days out of 10 have more than 4000 rentals.
combinations_9 = find_outcome_combinations(10, 9)

print(combinations_8)
print(combinations_9)




# The Probability For Each Combination
"""
p is the probability that an outcome will occur, and q is the complementary probability that the outcome won't happen -- 1−p=q1−p=q. """




# Calculating The Probability Of One Combination
prob_combination_3 = None

# Find the probability of a single combination for finding 3 days out of 5 are sunny
prob_combination_3 = .7 * .7 * .7 * .3 * .3
print(prob_combination_3)





# Function To Calculate The Probability Of A Single Combination
# find the probability that within 10 days, 7 or more days have more than 4000 riders.
# The probability of having more than 4000 riders on any single day is about .6. This means that pp is .6, and qq is .4
# Write a function to find the probability of a single combination occuring.
def single_combination(day1, day2):
    prob = (p ** day2) * (q ** (day1-day2))
    return prob

p = .6
q = .4

# Use the function to calculate the probability of 8 days out of 10 having more than 4000 riders
prob_8 = single_combination(10, 8) * find_outcome_combinations(10, 8)

# Use the function to calculate the probability of 9 days out of 10 having more than 4000 riders.
prob_9 = single_combination(10, 9) * find_outcome_combinations(10, 9)

# Use the function to calculate the probability of 10 days out of 10 having more than 4000 riders
prob_10 = single_combination(10, 10) * find_outcome_combinations(10, 10)

print(prob_8)
print(prob_9)
print(prob_10)





# Statistical Significance
"""
We touched on the question of statistical significance before
-- it's the question of whether a result happened as the result of something we changed,
or whether a result is a matter of random chance.

Typically, researchers will use 5% as a significance threshold --
if an event would only happen 5% or less of the time by random chance, then it is statistically significant."""
"""
Let's say we've invented a weather control device that can make the weather sunny (if only!),
and we decide to test it on Los Angeles. The device isn't perfect, and can't make every single day sunny
-- it can only increase the chance that a day is sunny. We turn it on for 10 days, and notice that the weather is sunny in 8 of those.

In our case, there is 12% chance that the weather would be sunny 8 days out of 10 by random chance.
We add this to 4% for 9 days out of 10, and .6% for 10 days out of 10 to get a 16.6% total chance of the sunny outcome
happening 8 or more time in our 10 days. Our result isn't statistically significant,
so we'd have to go back to the lab and spend some time adding more flux capacitors to our weather control device.

Let's say we recalibrate our weather control device successfully, and observe for 10 more days, of which 9 of them are sunny.
This only has a 4.6% chance of happening randomly (probability of 9 plus probability of 10).
This is a statistically significant result, but it isn't a slam-dunk.
It would require more investigation, including collecting results for more days, to get a more conclusive result."""