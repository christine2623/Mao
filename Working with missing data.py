"""
In Python, we have the None keyword and type, which indicates no value.
The Pandas library uses NaN, which stands for "not a number", to indicate a missing value.
"""

# Using the as keyword assigns the import to a different name, so we can reference it more easily
# In this case, instead of having to type pandas all the time, we can just type pd
import pandas as pd

# Read in the survival data
f = "titanic_survival.csv"
titanic_survival = pd.read_csv(f)

# Print out the age column
print(titanic_survival["age"])

# We can use the isnull function to find which values in a column are missing
# age_null is a boolean vector, and has "True" where age is NaN, and "False" where it isn't
age_null = pd.isnull(titanic_survival["age"])

# Count the number of null values in the "age" column.
age_null_count = 0
for row in age_null:
    if row == True:
        age_null_count += 1






import pandas as pd
mean_age = sum(titanic_survival["age"]) / len(titanic_survival["age"])
# Unfortunately, mean_age is NaN.  This is because any calculations we do with a null value also result in a null value.
# This makes sense when you think about it -- how can you add a null value to a known value?
print(mean_age)

# What we have to do instead is filter the missing values out before we compute the mean.
age_null = pd.isnull(titanic_survival["age"])
new_age = titanic_survival["age"][age_null == False]

correct_mean_age = sum(new_age) / len(new_age)





# We can use the .mean() method to compute the mean, and it will automatically remove missing values.
import pandas as pd

# This is the same value that we computed in the last screen, but it's much simpler.
# The ease of using the .mean() method is great, but it's important to understand how the underlying data looks.
correct_mean_age = titanic_survival["age"].mean()
correct_mean_fare = titanic_survival["fare"].mean()






# the dictionary fares_by_class should have 1, 2, and 3 as keys, with the average fares as the corresponding values.
passenger_classes = [1, 2, 3]
fares_by_class = {}
for classes in passenger_classes:
    # Select just the rows in titanic_survival where the pclass value is equivalent to the current iterator value (age group)
    classes_row = titanic_survival[titanic_survival["pclass"] == classes]
    # Calculate mean value of this column using the Series method mean
    mean_fare = classes_row["fare"].mean()
    # In the fares_by_class dictionary, assign the calculated mean as the value and the class as the key
    fares_by_class[classes] = mean_fare






# The pivot_table method borrows it's name from pivot tables in Excel and it works in a similar way

import numpy as np

# Let's compute the survival change from 0-1 for people in each class
# The closer to one, the higher the chance people in that passenger class survived
# The "survived" column contains a 1 if the passenger survived, and a 0 if not
# The pivot_table method on a pandas dataframe will let us do this
# index specifies which column to subset data based on (in this case, we want to compute the survival percentage for each class)
# values specifies which column to subset based on the index
# The aggfunc specifies what to do with the subsets
# In this case, we split survived into 3 vectors, one for each passenger class, and take the mean of each
passenger_survival = titanic_survival.pivot_table(index="pclass", values="survived", aggfunc=np.mean)

# First class passengers had a much higher survival chance
print(passenger_survival)

# Use the pivot_table method to compute the mean "age" for each passenger class ("pclass").
passenger_age = titanic_survival.pivot_table(index="pclass", values="age", aggfunc=np.mean)
print(passenger_age)







import numpy as np

# This will compute the mean survival chance and the mean age for each passenger class
passenger_survival = titanic_survival.pivot_table(index="pclass", values=["age", "survived"], aggfunc=np.mean)
print(passenger_survival)

# Make a pivot table that computes the mean "age", survival chance("survived"), and "fare" for each embarkation port ("embarked")
port_stats = titanic_survival.pivot_table(index="embarked", values=["age", "survived", "fare"], aggfunc=np.mean)
print(port_stats)






# We can use the dropna method on Pandas dataframes to remove missing values in a matrix
# Using the method will drop any rows that contain missing values.

import pandas as pd

# Drop all rows that have missing values
new_titanic_survival = titanic_survival.dropna()

# It looks like we have an empty dataframe now.
# This is because every row has at least one missing value.
print(new_titanic_survival)

# We can also use the axis argument to drop columns that have missing values
new_titanic_survival = titanic_survival.dropna(axis=1)
print(new_titanic_survival)

# We can use the subset argument to only drop rows if certain columns have missing values.
# This drops all rows where "age" or "sex" is missing.
new_titanic_survival = titanic_survival.dropna(subset=["age", "sex"])

# Drop all rows in titanic_survival where the columns "age", "body", or "home.dest" have missing values.
new_titanic_survival = titanic_survival.dropna(subset=["age", "body", "home.dest"])






# In Pandas, dataframes and series have row indices.

# See the numbers to the left of each row?
# Those are row indexes.
# Since the data has so many columns, it is split into multiple lines, but there are only 5 rows.
print(titanic_survival.iloc[:5,:])


new_titanic_survival = titanic_survival.dropna(subset=["body"])
# Now let's print out the first 5 rows in new_titanic_survival
# The row indexes here aren't the same as in titanic_survival
# This is because we modified the titanic_survival dataframe to generate new_titanic_survival
# The row indexes you see here are the rows from titanic_survival that made it through the dropna method (didn't have missing values in the "body" column)
# They retain their original numbering, though
print(new_titanic_survival.iloc[:5,:])

# We've been using the .iloc method to address rows and columns
# .iloc works by position (row/column number)

# This code prints the fourth row in the data
print(new_titanic_survival.iloc[3,:])

# Using .loc instead addresses rows and columns by index, not position
# This actually prints the first row, because it has index 3
print(new_titanic_survival.loc[3,:])

# Assign the row with index 25 to row_index_25
row_index_25 = new_titanic_survival.loc[25,:]
# Assign the fifth row to row_position_fifth.
row_position_fifth = new_titanic_survival.iloc[4,:]






new_titanic_survival = titanic_survival.dropna(subset=["body"])

# This prints the value in the first column of the first row
print(new_titanic_survival.iloc[0,0])

# This prints the exact same value -- it prints the value at row index 3 and column "pclass"
# This happens to also be at row 0, index 0
print(new_titanic_survival.loc[3,"pclass"])

# Assign the value at row index 1100, column index "age" to row_1100_age.
row_1100_age = new_titanic_survival.loc[1100, "age"]
# Assign the value at row index 25, column index "survived" to row_25_survived
row_25_survived = (new_titanic_survival.loc[25, "survived"])






"""Sometimes it is useful to reindex, and make new indices starting from 0.
To do this, we can use the reset_index() method"""

# The indexes are the original numbers from titanic_survival
new_titanic_survival = titanic_survival.dropna(subset=["body"])
print(new_titanic_survival)

# Reset the index to an integer sequence, starting at 0.
# The drop keyword argument specifies whether or not to make a dataframe column with the index values.
# If True, it won't, if False, it will.
# We'll almost always want to set it to True.
new_titanic_survival = new_titanic_survival.reset_index(drop=True)
# Now we have indexes starting from 0!
print(new_titanic_survival)

# Use the dropna method to remove rows from titanic_survival that have missing values in the "age" or "boat" columns.
new_titanic_survival2 = titanic_survival.dropna(subset=["age", "boat"])
# Then, reindex the resulting dataframe so the row indices start from 0.
titanic_reindexed = new_titanic_survival2.reset_index(drop = True)








"""By default, .apply() will iterate through each column in a dataframe, and perform a function on it.
The column will be passed into the function.
The result from the function will be combined with all of the other results, and placed into a new series.
The function results will have the same position as the column they were generated from."""

import pandas as pd

# Let's look at a simple example.
# This function counts the number of null values in a series
def null_count(column):
    # Make a vector that contains True if null, False if not.
    column_null = pd.isnull(column)
    # Create a new vector with only values where the series is null.
    null = column[column_null == True]
    # Return the count of null values.
    return len(null)

# Compute null counts for each column
column_null_count = titanic_survival.apply(null_count)
print(column_null_count)

# Write a function to count up the number of non-null elements in a series.
def non_null_count(column):
    column_null = pd.isnull(column)
    non_null = column[column_null == False]
    return len(non_null)

# Use the .apply() method, along with your function, to run across all the columns in titanic_survival
column_not_null_count = titanic_survival.apply(non_null_count)
print(column_not_null_count)





# By passing in the axis argument, we can use the .apply() method to iterate over rows instead of columns.

# This function will check if a row is an entry for a minor (under 18), or not.
def is_minor(row):
    if row["age"] < 18:
        return True
    else:
        return False

# This is a boolean series with the same length as the number of rows in titanic_survival
# Each entry is True if the row at the same position is a record for a minor
# The axis of 1 specifies that it will iterate over rows, not columns
minors = titanic_survival.apply(is_minor, axis=1)



import pandas as pd
# If someone is under 18, they are a "minor". If they are over 18, they are an "adult".
# If their age is missing (is null), their age is "unknown"
def generate_age_label(row):
    age = row["age"]
    if pd.isnull(age):
        return "unknown"
    elif age < 18:
        return "minor"
    else:
        return "adult"
# use the function along with .apply() to find the correct label for everyone.
age_labels = titanic_survival.apply(generate_age_label, axis=1)






# Make a pivot table that computes the mean survival chance("survived"), for each age group ("age_labels")
age_group_survival = titanic_survival.pivot_table(index="age_labels", values="survived", aggfunc=np.mean)

