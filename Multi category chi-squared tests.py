# Multiple Categories
# apply the chi-squared test to figure out if there's a statistically significant correlation between two categorical columns.




# Calculating Expected Values
"""
In the single category chi-squared test, we find expected values from other data sets, and then compare with our own observed values.
In a multiple category chi-squared test, we calculate expected values across our whole dataset.
"""
"""
We can use our total proportions to calculate expected values.
24.1% of all people in income earn >50k, and 33.1% of all people in income are Female,
so we'd expect the proportion of people who are female and earn >50k to be .241 * .331, which is .0799771.
We have this expectation based on the proportions of Females and >50k earners across the whole dataset.
Instead, we see that the observed proportion is .036, which indicates that
there may be some correlation between the sex and high_income columns."""
# calculate the expected values
# Calculate the expected value for Males who earn >50k, and assign to males_over50k.
males_over50k = 0.669 * 0.241 * 32561
# Calculate the expected value for Males who earn <=50k, and assign to males_under50k.
males_under50k = 0.669 * 0.759 * 32561
# Calculate the expected value for Females who earn >50k, and assign to females_over50k.
females_over50k = 0.331 * 0.241 * 32561
# Calculate the expected value for Females who earn <=50k, and assign to females_under50k
females_under50k = 0.331 * 0.759 * 32561

print(males_over50k)
print(males_under50k)
print(females_over50k)
print(females_under50k)




# Calculating Chi-Squared
"""
Subtract the expected value from the observed value.
Square the difference.
Divide the squared difference by the expected value.
Repeat for all the observed and expected values and add up the the values
"""
# Compute the chi-squared value for the observed values above and the expected values above.
expect = [5249.8, 16533.5, 2597.4, 8180.3]
observe = [6662, 15128, 1179, 9592]
diff_list = []
for i, exp in enumerate(expect):
    diff = ((observe[i] - exp)**2)/exp
    diff_list.append(diff)
chisq_gender_income = sum(diff_list)
print(chisq_gender_income)




# Finding Statistical Significance
from scipy.stats import chisquare
chisquare, pvalue_gender_income = chisquare(observe, expect)
print(pvalue_gender_income)





# Cross Tables
# look at sex vs race
"""
Before we can find the chi-squared value, we need to find the observed frequency counts.
We can do this using the pandas.crosstab function.
The crosstab function will print a table that shows frequency counts for two or more columns.
"""
"""Example
import pandas
table = pandas.crosstab(income["sex"], [income["high_income"]])
print(table)"""
import pandas
income = pandas.read_csv("income_person.csv")

table = pandas.crosstab(income["sex"], [income["race"]])
print(table)



# Finding Expected Values
"""
We can use the scipy.stats.chi2_contingency function to generate the expected values.
The function takes in a cross table of observed counts, and returns the chi-squared value,
the p-value, the degrees of freedom, and the expected frequencies.
"""
""" Example
import numpy as np
from scipy.stats import chi2_contingency
observed = np.array([5, 5], [10, 10])
chisq_value, pvalue, df, expected = chi2_contingency(observed)"""
# Use the scipy.stats.chi2_contingency function to calculate the pvalue for the sex and race columns of income.
import numpy as np
from scipy.stats import chi2_contingency
table = pandas.crosstab(income["sex"], [income["race"]])
# [119, 346 ...] is the first row in the table, [192, 693...] is the second row
# another way: observed = np.array([119, 346, 1555, 109, 8642], [192, 693, 1569, 162, 19174])
chi_square, pvalue_gender_race, df, expected = chi2_contingency(table)
print(chi_square)
print(pvalue_gender_race)
print(df)
print(expected)





# Caveats
"""
There are a few caveats to using the chi-squared test that are important to cover:

Finding that a result isn't significant doesn't mean that no association between the columns exists.
For instance, if we found that the chi-squared test between the sex and race columns returned a p-value of .1,
it wouldn't mean that there is no relationship between sex and race.
It just means that there isn't a statistically significant relationship.

Finding a statistically significant result doesn't imply anything about what the correlation is.
For instance, finding that a chi-squared test between sex and race results in a p-value of .01
doesn't mean that the dataset contains too many Females who are White (or too few).
A statistically significant finding means that some evidence of a relationship between the variables exists,
but needs to be investigated further.

Chi-squared tests can only be applied in the case where each possibility within a category is independent.
For instance, the Census counts individuals as either Male or Female, not both.

Chi-squared tests are more valid when the numbers in each cell of the cross table are larger.
So if each number is 100, great -- if each number is 1, you may need to gather more data."""
