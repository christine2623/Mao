world_alcohol = [
                     [1986, "Western Pacific", "Viet Nam", "Wine", 0],
                     [1986, "Americas", "Uruguay", "Other", 0.5],
                     [1986, "Africa", "Cte d'Ivoire", "Wine", 1.62]
                    ]

#  we wanted to compute the average of the Display Value column
# Extract the values in the 5th column.
liters_drank = []
for row in world_alcohol:
    liters = row[4]
    liters_drank.append(liters)
liters_drank = liters_drank[1:len(liters_drank)]

# Calculate the average of the values in the 5th column.
total = 0
for item in liters_drank:
    total = total + float(item)
average = total / len(liters_drank)
print(average)






# Use the csv module to read world_alcohol.csv into the variable world_alcohol.
"""import csv
a = open("world_alcohol.csv", "r")"""
# world_alcohol should be a list of lists.
"""world_alcohol = list(csv.reader(a))"""

# Extract the first column of world_alcohol, and assign it to the variable years.
years = []
for row in world_alcohol:
    year = row[0]
    years.append(year)
# Use list slicing to remove the first item in years (this is a header)
years = years[1:]
# Find the sum of all the items in years. Assign the result to total.
total = 0
for items in years:
    total += float(items)
# Divide total by the length of years to get the average. Assign the result to avg_year.
avg_year = total / len(years)





"""Using NumPy, we can much more efficiently analyze data than we can using lists.
NumPy is a Python module that is used to create and manipulate multidimensional arrays."""
"""An array is a collection of values. Arrays have one or more dimensions.
An array dimension is the number of indices it takes to extract a value.
In a list, we specify a single index, so it is one-dimensional"""

# A list is similar to a NumPy one-dimensional array, or vector
# This is a two-dimensional array, also known as a matrix

"""To get started with NumPy, we first need to import it using import numpy.
We can then read in datasets using the genfromtxt() method."""
# Since world_alcohol.csv is a csv file, rows are separated by line breaks, and columns are separated by commas,
"""In files like this, the comma is called the delimiter, because it indicates where each field ends and a new one begins.
Other delimiters, such as tabs, are occasionally used, but commas are the most common."""

"""To use the genfromtxt(), we need to pass a keyword argument called delimiter that indicates what character is the delimiter"""
"""import numpy
nfl = numpy.genfromtxt("nfl.csv", delimiter=",")"""



# We can directly construct arrays from lists using the array() method.

import numpy
# When we input a list of lists, we get a matrix as a result:
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])

# We can use the shape property on arrays to figure out how many elements are in an array.
# For vectors, the shape property contains a tuple with 1 element.
# A tuple is a kind of list where the elements can't be changed.

vector = numpy.array([1, 2, 3, 4])
print(vector.shape)


"""For matrices, the shape property contains a tuple with 2 elements.
matrix = numpy.array([[5, 10, 15], [20, 25, 30]])
print(matrix.shape)
The above code will result in the tuple (2,3) indicating that matrix has 2 rows and 3 columns."""


# Each value in a NumPy array has to have the same data type.
# You can check the data type of a NumPy array using the dtype property

"""Because all of the values in a NumPy array have to have the same data type,
NumPy attempted to convert all of the columns to floats when they were read in.
The genfromtxt() method will attempt to guess the correct data type of the array it creates."""

"""When NumPy can't convert a value to a numeric data type like float or integer,
it uses a special nan value that stands for Not a Number.
NumPy assigns an na value, which stands for Not Available, when the value doesn't exist. nan and na values are types of missing data."""


"""Scientific notation is a way to condense how very large or very precise numbers are displayed.
We can represent 100 in scientific notation as 1e+02.
The e+02 indicates that we should multiply what comes before it by 10 ^ 2(10 to the power 2, or 10 squared).
This results in 1 * 100, or 100. Thus, 1.98600000e+03 is actually 1.986 * 10 ^ 3, or 1986."""
# NumPy displays numeric values in scientific notation by default to account for larger or more precise numbers.
"""Specifying the keyword argument dtype when reading in world_alcohol.csv, and setting it to "U75".
This specifies that we want to read in each value as a 75 byte unicode data type."""
"""Specifying the keyword argument skip_header, and setting it to True.
This will skip the first row of world_alcohol.csv when reading in the data."""


# Use the NumPy function genfromtxt to read in world_alcohol.csv
# Set the dtype parameter to "U75".
# Set the skip_header parameter to True.
# Set the delimiter parameter to ,.
world_alcohol = numpy.genfromtxt("world_alcohol.csv", delimiter = ",", dtype = "U75", skip_header = True)
print(world_alcohol)



# Assign the amount of alcohol Uruguayans drank in other beverages per capita in 1986 to uruguay_other_1986.
# This is the second row and fifth column.
uruguay_other_1986 = world_alcohol[1, 4]
# Assign the country in the third row to third_country. Country is the third column.
third_country = world_alcohol[2, 2]




"""select all of the rows, but only the column with index 1.
So we'll end up with 10, 25, 40, which is the whole second column."""
# The colon by itself : specifies that the entirety of a single dimension should be selected
matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix[:,1])


# Assign the whole third column from world_alcohol to the variable countries.
countries = world_alcohol[:, 2]
# Assign the whole fifth column from world_alcohol to the variable alcohol_consumption.
alcohol_consumption = world_alcohol[:, 4]





# select one whole dimension, and a slice of the other
"""matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix[:,0:2])"""



"""We can select rows by specifying a colon in the columns area:
print(matrix[1:3,:])
"""


"""We can also select a single value alongside an entire dimension:
print(matrix[1:3,1])
"""


# Assign all the rows and the first 2 columns of world_alcohol to first_two_columns.
first_two_columns = world_alcohol[:,0:2]
# Assign the first 10 rows and the first column of world_alcohol to first_ten_years.
first_ten_years = world_alcohol[0:10, 0]
# Assign the first 10 rows and all of the columns of world_alcohol to first_ten_rows
first_ten_rows = world_alcohol[0:10, :]


# Assign the first 20 rows of the columns at index 1 and 2 of world_alcohol to first_twenty_regions
first_twenty_regions = world_alcohol[0:20, 1:3]







