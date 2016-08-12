"""One of the most powerful aspects of NumPy is the ability to make comparisons across an entire array.
These comparisons result in Boolean values.
Here's an example with a vector:

vector = numpy.array([5, 10, 15, 20])
vector == 10

If you'll recall from an earlier mission, the double equals sign (==) compares two values.
 In the case of NumPy, the second value will be compared to each element in the vector.
 If the value are equal, True will be returned, and False otherwise.
The above code will result in the vector [False, True, False, False]. This is because only the second element in vector equals 10."""


"""matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
    matrix == 25
This will compare 25 to every element in matrix. The result will be a matrix where elements are True or False"""



# Extract the third column in world_alcohol, and compare the column with the string Canada.
# Assign the result to countries_canada.
countries_canada = (world_alcohol[:, 2]== "Canada")
# Extract the first column in world_alcohol, and compare the column with the string 1984.
# Assign the result to years_1984
years_1984 = (world_alcohol[:, 0]== "1984")





"""vector = numpy.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
# We use equal_to_ten to only select elements in vector where equal_to_ten is True. This results in the vector [10].
print(vector[equal_to_ten])"""



matrix = numpy.array([
                [5, 10, 15],
                [20, 25, 30],
                [35, 40, 45]
             ])
# We compare the second column of matrix with the value 25.
# This results in a vector [False, True, False], which we assign to second_column_25
second_column_25 = (matrix[:,1] == 25)
    # We use second_column_25 to select any rows in matrix where second_column_25 is True
print(matrix[second_column_25, :])
"""We end up with the matrix below:
[
    [20, 25, 30]
]"""



# Compare the third column of world_alcohol to the string Algeria.
country_is_algeria = (world_alcohol[:, 2] == "Algeria")
# elect only the rows in world_alcohol where country_is_algeria is True.
country_algeria = world_alcohol[country_is_algeria, :]




"""We can also perform comparisons with multiple conditions.
When we do this, we specify each condition separately, then join them with the & symbol."""
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_and_five = (vector == 10) & (vector == 5)
"""By using the & symbol, we indicate that both conditions must be True for the final result to be True.
The statement returns [False, False, False, False], because no elements can simultaneously be 10 and 5"""

# We can also use the | symbol to specify that one condition or the other should be True




# Compare the first column of world_alcohol to the string 1986
# Compare the third column of world_alcohol to the string Algeria
# join the conditions with &
is_algeria_and_1986 = (world_alcohol[:,0] == "1986")&(world_alcohol[:,2] == "Algeria")
# Use is_algeria_and_1986 to select rows from world_alcohol.
rows_with_algeria_and_1986 = world_alcohol[is_algeria_and_1986]




# Another powerful use for comparisons is replacing values in an array.
matrix = numpy.array([
            [5, 10, 15],
            [20, 25, 30],
            [35, 40, 45]
         ])
# compare the second column of matrix to the value 25
# select any values in the second column of matrix where the value is 25
second_column_25 = matrix[:,1] == 25
# replace the selected values with 10
matrix[second_column_25, 1] = 10




# Replace all instances of the string 1986 in the first column of world_alcohol with the string 2014.
a = (world_alcohol[:,0] == "1986")
world_alcohol[a, 0] = "2014"
# Replace all instances of the string Wine in the fourth column of world_alcohol with the string Grog.
b= (world_alcohol[:,3]=="Wine")
world_alcohol[b,3] = "Grog"





# deal with empty string values ('')
# Compare all the items in the fifth column of world_alcohol with an empty string ''. Assign the result to is_value_empty
is_value_empty = (world_alcohol[:, 4] == '')
# Select all the values in the fifth column of world_alcohol where is_value_empty is True, and replace them with the string 0
world_alcohol[is_value_empty, 4] = '0'





# Extract the fifth column from world_alcohol, and assign it to the variable alcohol_consumption.
alcohol_consumption = world_alcohol[:, 4]
# Use the astype() method to convert alcohol_consumption to the float data type.
alcohol_consumption = alcohol_consumption.astype(float)






"""sum() -- computes the sum of all the elements in a vector, or the sum along a dimension in a matrix.
mean() -- computes the average of all the elements in a vector, or the average along a dimension in a matrix.
max() -- computes the maximum of all the elements in a vector, or the maximum along a dimension in a matrix."""

"""for sum()
With a matrix, we have to specify an additional keyword argument axis.
The axis dictates which dimension we perform the operation on.
1 means that we want to perform the operation on each row, and 0 means on each column. """

"""  matrix = numpy.array([
                [5, 10, 15],
                [20, 25, 30],
                [35, 40, 45]
             ])
    matrix.sum(axis=1)"""

# Use the sum() method to calculate the sum of the values in alcohol_consumption. Assign the result to total_alcohol.
total_alcohol = alcohol_consumption.sum()
# Use the mean() method to calculate the average of the values in alcohol_consumption. Assign the result to average_alcohol
average_alcohol = alcohol_consumption.mean()






# Create a matrix called canada_1986
# that contains only rows in world_alcohol
# where the first column is the string 1986 and the third column is the string Canada.
is_canada_1986 = (world_alcohol[:,0] == "1986") & (world_alcohol[:,2] == "Canada")
canada_1986 = world_alcohol[is_canada_1986, :]
# Extract the fifth column of canada_1986
canada_alcohol = canada_1986[:,4]
# replace any empty strings ('') with the string 0
empty_strings= (canada_alcohol == '')
canada_alcohol[empty_strings] = "0"
# convert the column to the float data
canada_alcohol = canada_alcohol.astype(float)
# Compute the sum of canada_alcohol. Assign the result to total_canadian_drinking
total_canadian_drinking = canada_alcohol.sum()





# calculate consumption for all countries in a given year.
# Create an empty dictionary called totals
totals = {}
# Select only the rows in world_alcohol that match a given year.
years = (world_alcohol[:,0] == "1989")
year = world_alcohol[years,:]
# Loop through a list of countries
for row in countries:
    # Select only the rows from year that match the given country
    is_country = (year[:,2] == row)
    country_consumption = year[is_country, :]
    # Extract the fifth column from country_consumption
    alcohol_column = country_consumption[:, 4]
    # Replace any empty string values in the column with the string 0
    empty_string = (alcohol_column == '')
    alcohol_column[empty_string] = "0"
    # Convert the column to the float data type
    alcohol_column = alcohol_column.astype(float)
    # Find the sum of the column
    # Add the sum to the totals dictionary with the country name as the key.
    totals[row] = alcohol_column.sum()





totals = {'Afghanistan': 0.0,
 'Albania': 1.73,
 'Algeria': 0.40000000000000002,
 'Angola': 2.2799999999999998,
 'Antigua and Barbuda': 4.6900000000000004,
 'Argentina': 10.82,
 'Australia': 12.09,
 'Austria': 13.9,
 'Bahamas': 12.290000000000001,
 'Bahrain': 4.8899999999999997,
 'Bangladesh': 0.0,
 'Belarus': 7.9799999999999995,
 'Belgium': 11.609999999999999}

# find the key with the highest value
# Create a variable called highest_value to keep track of the highest value. Set it to 0 initially
highest_value = 0
# Create a variable called highest_key to keep track of the key associated with the highest value. Set it to None initially.
highest_key = None
# Loop through each key in the dictionary
for key in totals:
    # If the value associated with the key is greater than highest_value,
    # assign the value to highest_value, and assign the key to highest_key
    if highest_value < totals[key]:
        highest_value = totals[key]
        highest_key = key

print(highest_value)
print(highest_key)



"""NumPy is much easier to work with than lists of lists:

It's easy to perform computations on data.
Data indexing and slicing is faster and easier.
Data types can be converted quickly.
Overall, NumPy makes working with data in Python much more efficient. 
NumPy is widely used in a variety of tasks, particularly machine learning, due to this efficiency.

However, you may have noticed some limitations with NumPy as you worked through the past 2 missions:

A whole array has to be the same datatype, which makes it cumbersome to work with many datasets.
Columns and rows have to be referred to by number, which gets confusing when you go back and forth from column name to column number.
In the next few missions, we'll learn about the Pandas library, one of the most popular data analysis libraries.
Pandas builds on NumPy but does a better job of addressing the limitations of NumPy."""