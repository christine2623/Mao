"""Pandas has quickly became an important tool in a data professional's toolbelt and is the most popular library for working with tabular data in Python.
Tabular data is any data that can be represented as rows and columns. """

"""To represent tabular data, Pandas uses a custom data structure called a DataFrame.
A DataFrame is a highly efficient, 2-dimensional data structure that provides a suite of methods and attributes
to quickly explore, analyze, and visualize data.
The DataFrame object is similar to the NumPy 2D array
but adds support for many features that help you work with tabular data."""

# One of the biggest advantages that Pandas has over NumPy is the ability to store mixed data types in rows and columns.
# Pandas DataFrames can also handle missing values gracefully using a custom object, NaN, to represent those values.




# Import the Pandas library.
import pandas
# Use the Pandas function read_csv() to read the file "food_info.csv" into a DataFrame named food_info.
food_info = pandas.read_csv("food_info.csv")
# Use the type() and print() functions to display the type of food_info to confirm that it's a DataFrame object.
print(type(food_info))




"""To select the first 5 rows of a DataFrame, use the DataFrame method head().
When you call the head() method, Pandas will return a new DataFrame containing just the first 5 rows"""
# first_rows = food_info.head()
# you can pass in an integer (n) into the head() method to display the first n rows instead of the first 5
# To access the full list of column names, use the columns attribute
"""you can use the shape attribute to understand the dimensions of the DataFrame.
The shape attribute returns a tuple of integers representing the number of rows followed by the number of columns"""
"""# Returns the tuple (8618,36) and assigns to `dimensions`.
dimensions = food_info.shape
# The number of rows, 8618.
num_rows = dimensions[0]
# The number of columns, 36.
num_cols = dimensions[1]"""


# Select the first 20 rows from food_info and assign to the variable first_twenty.
first_twenty = food_info.head(20)
print(food_info.head(3))
dimensions = food_info.shape
print(dimensions)
num_rows = dimensions[0]
print(num_rows)
num_cols = dimensions[1]
print(num_cols)







"""The Series object is a core data structure that Pandas uses to represent rows and columns.
A Series is a labelled collection of values similar to the NumPy vector.
The main advantage of Series objects is the ability to utilize non-integer labels.
NumPy arrays can only utilize integer labels for indexing.

Pandas utilizes this feature to provide more context when returning a row or a column from a DataFrame.
For example, when you select a row from a DataFrame, instead of just returning the values in that row as a list,
Pandas returns a Series object that contains the column labels as well as the corresponding values"""




"""While we use bracket notation to access elements in a NumPy array or a standard list,
we need to use the Pandas method loc[] to select rows in a DataFrame.
The loc[] method allows you to select rows by row labels.
Recall that when you read a file into a DataFrame, Pandas uses the row number (or position) as each row's label.
Pandas uses zero-indexing, so the first row is at index 0, the second row at index 1, and so on"""

# Series object representing the row at index 0.
food_info.loc[0]

# Assign the 100th row of food_info to the variable hundredth_row.
hundredth_row = food_info.loc[99]
# Display hundredth_row using the print() function.
print(hundredth_row)





"""The object dtype is equivalent to the string type in Python.
Pandas borrows from the NumPy type system and contains the following dtypes:

object - for representing string values.
int - for representing integer values.
float - for representing float values.
datetime - for representing time values.
bool - for representing Boolean values.

When reading a file into a DataFrame, Pandas analyzes the values and infers each column's types.
To access the types for each column, use the DataFrame attribute dtypes to return a Series
containing each column name and its corresponding type."""


print(food_info.dtypes)






"""If you're interested in accessing multiple rows of the DataFrame,
you can pass in either a slice of row labels or a list of row labels and Pandas will return a DataFrame object.
Note that unlike slicing lists in Python, a slice of a DataFrame using .loc[] will include both the start and the end row

# DataFrame containing the rows at index 3, 4, 5, and 6 returned.
food_info.loc[3:6]
# DataFrame containing the rows at index 2, 5, and 10 returned. Either of the following work.
# Method 1
two_five_ten = [2,5,10]
food_info.loc[two_five_ten]
# Method 2
food_info.loc[[2,5,10]]
"""

print("Rows 3, 4, 5 and 6")
print(food_info.loc[3:6])

print("Rows 2, 5, and 10")
two_five_ten = [2,5,10]
print(food_info.loc[two_five_ten])

# get the total number of rows
num_rows = food_info.shape[0]
# Select the last 5 rows of food_info and assign to the variable last_rows.
last_rows = food_info.loc[num_rows - 5:num_rows]






"""# Series object representing the "NDB_No" column.
ndb_col = food_info["NDB_No"]

# You can instead access a column by passing in a string variable.
col_name = "NDB_No"
ndb_col = food_info[col_name]"""

# Series object.
ndb_col = food_info["NDB_No"]
print(ndb_col)

# Display the type of the column to confirm it's a Series object.
print(type(ndb_col))

# Assign the "FA_Sat_(g)" column to the variable saturated_fat.
saturated_fat = food_info["FA_Sat_(g)"]
# Assign the "Cholestrl_(mg)" column to the variable cholesterol.
cholesterol = food_info["Cholestrl_(mg)"]





"""columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]

# Skipping the assignment.
zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]"""

zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]

columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]
# Select the 'Selenium_(mcg)' and 'Thiamin_(mg)' columns and assign the resulting DataFrame to selenium_thiamin
selenium_thiamin = food_info[['Selenium_(mcg)', 'Thiamin_(mg)']]







print(food_info.columns)
print(food_info.head(2))

# Select and display only the columns that use grams for measurement (that end with "(g)").
# Use the columns attribute to return the column names in food_info and convert to a list by calling the method tolist()
columns = food_info.columns.tolist()
# Create a new list, gram_columns
gram_columns = []
for items in columns:
    #  The string method endswith() returns True if the string object calling the method ends with the string passed into the parentheses.
    if items.endswith("(g)") == True:
        gram_columns.append(items)
# Pass gram_columns into bracket notation to select just those columns and assign the resulting DataFrame to gram_df
gram_df = food_info[gram_columns]
# Then use the DataFrame method head() to display the first 3 rows of gram_df.
print(gram_df.head(3))