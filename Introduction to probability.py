# Probability Basics
import pandas

flags = pandas.read_csv("flags.csv")
# Print the first two rows of the data.
print(flags[:2])

# Find the country with the most bars in its flag.
most_bars_country = flags["name"][flags["bars"].idxmax()]
print(most_bars_country)

# Find the country with the highest population (as of 1986).
highest_population_country = flags["name"][flags["population"].idxmax()]
print(highest_population_country)





# Calculating Probability
total_countries = flags.shape[0]

# Determine the probability of a country having a flag with the color orange in it.
orange_probability = flags[flags["orange"] == 1].shape[0]/total_countries

# Determine the probability of a country having a flag with more than 1 stripe in it.
stripe_probability = flags[flags["stripes"] > 1].shape[0]/total_countries

print(orange_probability)
print(stripe_probability)




# Conjunctive Probabilities
"""
let's say we have a coin that we flip 5 times, and we want to find the probability that it will come up heads every time.
This is called a conjunctive probability, because it involves a sequence of events."""

five_heads = .5 ** 5
# Find the probability that 10 flips in a row will all turn out heads.
ten_heads = .5 ** 10
# ind the probability that 100 flips in a row will all turn out heads.
hundred_heads = .5 ** 100

print(ten_heads)
print(hundred_heads)





# Dependent Probabilities
"""
Let's say that we're picking countries from the sample, and removing them when we pick.
Each time we pick a country, we reduce the sample size for the next pick.
The events are dependent -- the number of countries available to pick depends on the previous pick.
We can't just calculate the probability upfront and take a power in this case --
we need to recompute the probability after each selection happens."""

# we're picking countries from our dataset, and removing each one that we pick.
# What are the odds of picking three countries with red in their flags in a row?
# Remember that whether a flag has red in it or not is in the `red` column.
one_red = flags[flags["red"] == 1].shape[0]/total_countries
two_red = ((flags[flags["red"] == 1].shape[0]) - 1)/(total_countries - 1)
three_red = one_red * two_red * (((flags[flags["red"] == 1].shape[0]) - 2)/(total_countries - 2))

print(one_red)
print(two_red)
print(three_red)





# Disjunctive Probability
import random
start = 1
end = 18000
# we have a random number generator that generates numbers from 1 to 18000
def random_number_generator(times):
    number = [random.randint(times) for _ in range(18000)]
    return number

# What are the odds of getting a number evenly divisible by 100, with no remainder? (ie 100, 200, 300, etc).
hundred = [i for i in range(start,end+1) if i % 100 == 0]
hundred_prob = len(hundred)/end

# What are the odds of getting a number evenly divisible by 70, with no remainder?
seventy = [i for i in range(start,end+1) if i % 70 == 0]
seventy_prob = len(seventy)/end

print(hundred_prob)
print(seventy_prob)




# Disjunctive Dependent Probabilities
red_or_orange = None
stripes_or_bars = None

# Find the probability of a flag having red or orange as a color.
red = flags[flags["red"] == 1].shape[0]
orange = flags[flags["orange"] == 1].shape[0]
red_and_orange = flags[(flags["red"] == 1) & (flags["orange"] == 1)].shape[0]
red_or_orange = (red + orange - red_and_orange)/total_countries

# Find the probability of a flag having at least one stripes or at least one bars.
stripes = flags[flags["stripes"] > 0].shape[0]
bars = flags[flags["bars"] > 0].shape[0]
stripes_and_bars = flags[(flags["stripes"] > 0) & (flags["bars"] > 0)].shape[0]
stripes_or_bars = (stripes + bars - stripes_and_bars)/total_countries

print(red_or_orange)
print(stripes_or_bars)






# Disjunctive Probabilities With Multiple Conditions
heads_or = None

# Let's say we have a coin that we're flipping. Find the probability that at least one of the first three flips comes up heads.
prob_of_tails_top = 1/2
# Just have to calculate what if there is no head in the first three flips?
# Then perform total prob subtract all tails
heads_or = 1 - ((1/2) ** 3)

print(heads_or)

