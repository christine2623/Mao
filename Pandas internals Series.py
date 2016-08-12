"""The three key data structures in Pandas are:

Series (collection of values)
DataFrame (collection of Series objects)
Panel (collection of DataFrame objects)"""

"""Series objects use NumPy arrays for fast computation, but build on them by adding valuable features for analyzing data.
For example, while NumPy arrays utilize an integer index, Series objects can utilize other index types, like a string index.
Series objects also allow for mixed data types and utilize the NaN Python value for handling missing values."""

import pandas as pd
fandango = pd.read_csv("fandango_score_comparison.csv")
# use the .head() method to print the first 2 rows.
print(fandango.head(2))



fandango = pd.read_csv('fandango_score_comparison.csv')
series_film = fandango['FILM']
print(series_film.head(5))
series_rt = fandango['RottenTomatoes']
# print the first 5 values.
print(series_rt[0:5])



# Get rid of int index, use the film name(string) index instead
# Import the Series object from pandas
from pandas import Series
# get the .values of the film name column
film_names = series_film.values
rt_scores = series_rt.values
# Instantiate a new Series object, which takes in a data parameter and an index parameter
series_custom = Series(rt_scores , index=film_names)
series_custom[['Minions (2015)', 'Leviathan (2014)']]




"""Even though we specified that the Series object uses a custom, string index,
the object still maintains an internal integer index that we can use for selection.
In this way, Series objects act both like a dictionary and a list since we can access values using our custom index
(like the keys in a dictionary) or the integer index (like the index in a list)."""

series_custom = Series(rt_scores , index=film_names)
series_custom[['Minions (2015)', 'Leviathan (2014)']]
# Assign the values in series_custom from index 5 to index 10 to the variable fiveten
fiveten = series_custom[5:10]
print(fiveten)





# We can use the reindex() method to sort series_custom in alphabetical order by film.
# return a list representation of the current index using tolist()
original_index = series_custom.index.tolist()
# sort the index using sorted()
sorted_index = sorted(original_index)
# use reindex() to set the new ordered index
sorted_by_index = series_custom.reindex(sorted_index)





"""To make sorting easier, Pandas comes with a sort_index() method, which returns a Series sorted by the index,
and a sort_values() method method, which returns a Series sorted by the values.
Since the values representing the Rotten Tomatoes scores are integers,
sorting by values will sort in numerically ascending order (low to high) in our case."""
# Sort series_custom by index using sort_index() and assign to the variable sc2
sc2 = series_custom.sort_index()
# Sort series_custom by values and assign to the variable sc3
sc3 = series_custom.sort_values()
# print the first ten values
print(sc2[0:10])
print(sc3[0:10])




# series_custom > 50 will actually return a Series object with a boolean value for each film.
"""To retrieve the actual films a Series object containing just the films with a rating greater than 50,
we need to pass in this Boolean series into the original Series object.
series_greater_than_50 = series_custom[series_custom > 50]"""

criteria_one = series_custom > 50
criteria_two = series_custom < 75
# Return a filtered Series object that only contains the values where both criteria are true
both_criteria= series_custom[criteria_one & criteria_two]




# rt_critics and rt_users are Series objects, that contain the critics average rating and the users average rating for each film.
rt_critics = Series(fandango['RottenTomatoes'].values, index=fandango['FILM'])
rt_users = Series(fandango['RottenTomatoes_User'].values, index=fandango['FILM'])
# rt_mean, containing the mean of the critics and users rating for each film.
rt_mean = (rt_critics + rt_users)/2





