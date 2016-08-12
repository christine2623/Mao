import csv
# Create a File handler for nfl_suspensions_data.csv.
nfldata = open("nfl_suspensions_data.csv")
# Use the csv.reader and list methods to read the file into a list named nfl_suspensions.
nfl_suspensions_data = csv.reader(nfldata)
nfl_suspensions = list(nfl_suspensions_data)

# Remove the first list in nfl_suspensions since it's the header row
nfl_suspensions = nfl_suspensions[1:len(nfl_suspensions)]

# Count up the frequency for each value in the year column.
years = {}
for lists in nfl_suspensions:
    # Extract that row's value for the year column and assign to row_year.
    row_year = lists[5]
    # If row_year is already a key in years, add 1 to the value associated for that key.
    if row_year in years:
        years[row_year] += 1
    # If row_year isn't already a key in years, set the value associated with the key to 1.
    if row_year not in years:
        years[row_year] = 1

print(years)





# create a set of the unique values in the team column. Assign this set to unique_teams
unique_teams = set(row[1] for row in nfl_suspensions)
# create a set of the unique values in the games column. Assign this set to unique_games
unique_games = set(row[2] for row in nfl_suspensions)

print(unique_teams)
print(unique_games)





# Create a class named Suspension
class Suspension():
    # Sole required parameter is a list representing a row from the dataset
    def __init__(self, row):
        # Set the name value for that row to the name property
        self.name = row[0]
        # Set the team value for that row to the team property
        self.team = row[1]
        self.games = row[2]
        self.year = row[5]

# Create a Suspension instance using the third row in nfl_suspensions and assign to the variable third_suspension
third_suspension = Suspension(nfl_suspensions[2])




# Instead of assigning the value at index 5 to the year property directly, use a try except block
class Suspension():
    def __init__(self,row):
        self.name = row[0]
        self.team = row[1]
        self.games = row[2]
        # Tries to cast the value at index 5 to an integer
        try:
            self.year = int(row[5])
        # If an exception is thrown, assign the value 0 to the year property instead
        except Exception:
            self.year = 0

    # method get_year that returns the year value for that Suspension instance
    def get_year(self):
        return(self.year)

# Create a Suspension instance using the 23th row and assign to missing_year.
missing_year = Suspension(nfl_suspensions[22])
# Using the .get_year() method, assign the year of the missing_year suspension instance to twenty_third_year.
twenty_third_year = missing_year.get_year()