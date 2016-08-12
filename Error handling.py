"""A set is a data type where each element is unique.
A set behaves very similarly to a list, except if you add an element to a set that it already contains, it will ignore it.
A set also doesn't have an order to the items in it, unlike a list where each item as an index."""
# You can create a set by using the set() function.
"""unique_animals = set(["Dog", "Cat", "Hippo", "Dog", "Cat", "Dog", "Dog", "Cat"])
print(unique_animals)
We'll get {'Hippo', 'Cat', 'Dog'} as a result."""
# You can add items to the set using the add() method:  unique_animals.add("Tiger")
# You can remove items from a set using the remove() method

# Extract the gender column from legislators and get the unique value in gender
# create an empty list
gender = []
# looping and get the gender column
for rows in legislators:
    gender.append(rows[3])
# change the datatype to set() and get the unique values
gender = set(gender)
print(gender)



"""One of the most valuable things to do with a fresh dataset is to explore it and see if you can find any patterns. Patterns can be things like:
Missing data.
Sometimes, values will be completely missing, but other times, a string like N/A will be used to indicate missing values.
Values that don't make intuitive sense.
If one of our Congresspeople had a birthday in 2050, it wouldn't make much sense, and would be a problem with the data.
Recuring themes.
One theme in this dataset is that the overwhelming majority of Congresspeople are male."""



# Work with missing values
"""Remove any rows that contain missing data.
Fill the missing fields with a specified value.
Fill the missing fields with a calculated value.
Use analysis techniques that work with missing data."""
# Replace with specified value
"""rows = [
    ["Bassett", "Richard", "1745-04-02", "M", "sen", "DE", "Anti-Administration"],
    ["Bland", "Theodorick", "1742-03-21", "", "rep", "VA", ""]
]
for row in rows:
    if row[6] == "":
        row[6] = "No Party""""
for rows in legislators:
    if rows[3] == "":
        # replace the empty gender column to Male
        rows[3] = "M"



"""When data is in a format that's hard to work with, it's common to reformat the values to make them simpler.
In this case, we can split the date into components:
date = "1820-01-02"
parts = date.split("-")
print(parts)
"""
birth_years = []
for row in legislators:
    # Split the value in the birthday column on the - character
    parts = row[2].split('-')
    # Extract the first item in parts and append it to birth_years
    birth_years.append(parts[0])



# try/except block.
"""an empty string can't be converted to an integer. We can deal with errors with something known as a try/except block.
If you surround any code that causes an error with a try/except block, the error will be handled"""
"""try:
    int('')
except Exception:
    print("There was an error")"""
"""n the above code example, the Python interpreter will try to run int(''). This will cause a ValueError.
Instead of stopping the code from executing, it will be handled by the except statement,
which will print out There was an error.
The Python interpreter will continue to run any code lines that come after the except statement"""
# try to convert string hellp into float
try:
    float("hello")
except Exception:
    print("Error converting to float.")



"""When the Python interpreter creates an exception, it actually creates an instance of the Exception class.
This class has certain properties that allow us to debug the error.
We can access the instance of the Exception class in the except statement body:
try:
    int('')
except Exception as exc:
    print(type(exc))
"""
"""In the code above, we use the as statement to assign the instance of the Exception class to the variable exc.
We can then access the variable in the except statement body.
Printing type(exc) will print the type of Exception that occured in the try statement body."""
# attempts to convert an empty string to an integer
try:
    int('')
except Exception as exc:
    # Print the type of the Exception instance
    print(type(exc))
    # Convert the Exception instance to a string and print it out.
    print(str(exc))




# We dont want to do anything in the except block so we put "pass"
converted_years = []
# Try to convert years from string to integer
for element in birth_years:
    try:
        element = int(element)
    except Exception:
        # We dont want to do anything in the except block
        pass
    converted_years.append(element)



# create extra column which will be birth year in integer datatype
for row in legislators:
    # get the birthday column
    birthday = row[2]
    # split the string into list form and get the one with index[0]
    birth_year = birthday.split("-")[0]
    # try to convert the birth year into integer
    try:
        birth_year = int(birth_year)
    # set exception birth year value to 0
    except Exception:
        birth_year = 0
    # append the birth year back to legislators/row
    row.append(birth_year)


# in birth year column, replace any 0 values with the previous value.
last_value = 1
for row in legislators:
    if row[7] == 0:
        # If the year column (index 7) equals 0, replace it with last_value.
        row[7] = last_value
    # Assign the value of the year column (index 7) to last_value
    last_value = row[7]




