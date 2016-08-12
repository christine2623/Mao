import math
# sqrt() function inside the module takes a number as an argument, and returns the square root of that number.
a = math.sqrt(16.0)
# ceil() function returns the smallest integer that is greater than or equal to the input.
b = math.ceil(111.3)
# floor() function returns the largest integer less than or equal to the input.
c = math.floor(89.9)

print(a, b, c)



# Using the variable within a module
# There is a variable called pi in math module
a = math.sqrt(math.pi)
b = math.ceil(math.pi)
c = math.floor(math.pi)
print(math.pi)



# Import csv module for csv file processing
import csv
nfldata = open("nfl.csv")
# csv,reader() is a function returns an object that represents our data.
nfl = list(csv.reader(nfldata))



#Count how many games the "New England Patriots" won from 2009-2013
# First, import the csv module
import csv
# Then, open our file in `r` mode
f = open("nfl.csv", "r")
# Use the csv module to read the file, and convert the result to a list
nfl = list(csv.reader(f))

# Start our count at 0
patriots_wins = 0
# Loop through our dataset, counting the rows with "New England Patriots" as the winner
for row in nfl:
    if row[2] == "New England Patriots":
        patriots_wins += 1



# Define a function returns the number of wins the team had in the period covered by the dataset.
import csv

f = open("nfl.csv", 'r')
nfl = list(csv.reader(f))

def nfl_wins(team_name):
    wins = 0
    for rows in nfl:
        if rows[2] == team_name:
            wins += 1
    return wins

cowboys_wins = nfl_wins("Dallas Cowboys")
falcons_wins = nfl_wins("Atlanta Falcons")






# Create function that output the number of wins the team had in the given year, as an integer.
import csv

f = open("nfl.csv", 'r')
nfl = list(csv.reader(f))

def nfl_wins_in_a_year(teamname, year):
    count = 0
    for row in nfl:
        if row[2] == teamname and row[0] == year:
            count = count + 1
    return count

browns_2010_wins = nfl_wins_in_a_year("Cleveland Browns", "2010")
eagles_2011_wins = nfl_wins_in_a_year("Philadelphia Eagles", "2011")