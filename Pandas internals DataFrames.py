import pandas as pd
fandango = pd.read_csv("fandango_score_comparison.csv")
print(fandango.head(2))
# Use the index attribute to return the index of the DataFrame and display it using the print function.
print(fandango.index)




#In Series, each unique index value refers to a data value. In DataFrames, each index refers to an entire row.

fandango = pd.read_csv('fandango_score_comparison.csv')
first_row = 0
last_row = fandango.shape[0] - 1
# Return a DataFrame containing just the first and the last row and assign to first_last.
first_last = fandango.iloc[[first_row, last_row]]




"""The DataFrame object contains a set_index() method that allows you to pass in the name of the column
you'd like Pandas to use as the index for the DataFrame.
inplace: if set to True, will set the index to the current DataFrame instead of returning a new one
drop: if set to False, will keep the column you specified for the index in the DataFrame"""

fandango = pd.read_csv('fandango_score_comparison.csv')
# Use the Pandas DataFrame method set_index to assign the FILM column as the custom index for the DataFrame
# without the FILM column dropped from the DataFrame.
# We want to keep the original DataFrame so assign the new DataFrame to fandango_films.
fandango_films = fandango.set_index(fandango["FILM"], inplace = False, drop =False)
print(fandango_films.index)




# Selecting using custom index
"""# Slice using either bracket notation or loc[]
fandango_films["Avengers: Age of Ultron (2015)":"Hot Tub Time Machine 2 (2015)"]
fandango_films.loc["Avengers: Age of Ultron (2015)":"Hot Tub Time Machine 2 (2015)"]

# Specific movie
fandango_films.loc['Kumiko, The Treasure Hunter (2015)']

# Selecting list of movies
movies = ['Kumiko, The Treasure Hunter (2015)', 'Do You Believe? (2015)', 'Ant-Man (2015)']
fandango_films.loc[movies]"""

# Select just these movies in the following order from fandango_films
best_movies_ever = fandango_films.loc[["The Lazarus Effect (2015)", "Gett: The Trial of Viviane Amsalem (2015)", "Mr. Holmes (2015)"]]




# select only the float columns and calculate the standard deviation for the columns
import numpy as np

# returns the data types as a Series
types = fandango_films.dtypes
# filter data types to just floats, index attributes returns just column names
float_columns = types[types.values == 'float64'].index
# use bracket notation to filter columns to just float columns
float_df = fandango_films[float_columns]

# `x` is a Series object representing a column
deviations = float_df.apply(lambda x: np.std(x))

print(deviations)




# Use the apply() method to calculate the average of each movie's values for RT_user_norm and Metacritic_user_nom and assign to the variable rt_mt_means.
rt_mt_user = float_df[['RT_user_norm', 'Metacritic_user_nom']]
rt_mt_means = rt_mt_user.apply(lambda x: np.mean(x), axis = 1)
# print the first five values
print(rt_mt_means.head(5))
