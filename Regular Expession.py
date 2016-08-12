# A regular expression is a sequence of characters that describes a search pattern.

# In Python, we use the re module to work with regular expressions.

strings = ["bat", "robotics", "megabyte"]
# The special character "." is used to indicate that any character can be put in its place.
regex = "b.t"


# We can use "^" to match the start of a string and "$" to match the end of string.
""""^a" will match all strings that start with "a".
"a$" will match all strings that end with "a"."""


# Assign a regular expression that is 7 characters long and matches every string in strings, but not bad_string, to the variable regex
strings = ["better not put too much", "butter in the", "batter"]
bad_string = "We also wouldn't want it to be bitter"
regex = "^b.tter"



# dataset from top 1000 posts on AskReddit in 2015
"""Title -- The title of the post
Score -- The number of upvotes the post received
Time -- When the post was posted
Gold -- How much Reddit Gold was given to the post
NumComs -- Number of comments the post received"""

a = open("askreddit_2015.csv", "w")
c = a.write('What\'s your internet "white whale", something you\'ve been searching for years to find with no luck?')
d = a.write('11510')
e = a.write('1433213314.0')
f = a.write('1')
g = a.write('26195')
h = a.write("What's your favorite video that is 10 seconds or less?")
i = a.write('8656')
j = a.write('1434205517.0')
k = a.write('4')
l = a.write('8479')
m = a.write('What are some interesting tests you can take to find out about yourself?')
n= a.write('8480')
o= a.write('1443409636.0')
p= a.write('1')
q = a.write('4055')
r= a.write("PhD's of Reddit. What is a dumbed down summary of your thesis?")
s  = a.write('7927')
t = a.write('1440188623.0')
u = a.write('0')
v = a.write('13201')
w= a.write('What is cool to be good at, yet uncool to be REALLY good at?')
x= a.write('7711')
y = a.write('1440082910.0')
z = a.write('0')
aa = a.write('20325')


import csv
# Use the csv module to read and assign our dataset to posts_with_header
posts_with_header = open("askreddit_2015.csv", 'r')
b = list(csv.reader(posts_with_header))
# Use list slicing to exclude the first row, which represents the column names.
posts = b[1:]
# Use a for loop and string slicing to print the first 10 rows of the dataset.
for row in posts[0:10]:
    print(row)



# With re.search(regex, string), we can check if string is a match for regex
"""if re.search("needle", "haystack") != None:
    print("We found it!")
else:
    print("Not a match")
"""

# Import re module
import re

of_reddit_count = 0
for row in posts:
    # Count the number of posts in our dataset that match the regular expression "of Reddit"
    if re.search("of Reddit", row[0]) != None:
        of_reddit_count += 1



"""If you look closely at the dataset, you may notice that some posts use "of Reddit", whereas others use "of reddit".
The capitalization of "Reddit" is different, but these are still the same format.
We can account for this inconsistency using square brackets. """

"""the regular expression "[bcr]at" would be matched by strings with the substrings "bat", "cat", and "rat", but nothing else.
We indicate that the first character in the regular expression can be either a "b", "c" or "r""""

import re

of_reddit_count = 0
for row in posts:
    # account for both capitalizations of "Reddit" using square bracket notation
    if re.search("of [Rr]eddit", row[0]) != None:
        of_reddit_count += 1


# We use "\" (backslash) to escape characters in regular expressions.
"""Suppose we wanted to match all strings that end with a period.
If we used ".$", it would match all strings that contain any character, since "." has that special meaning.
Instead, we must escape the "." with a backslash, and our regular expression becomes "\.$""""

import re

serious_count = 0
for row in posts:
    # Escape the square bracket characters to count the number of posts in our dataset that contain the "[Serious]" tag
    if re.search("\[Serious\]", row[0]) != None:
        serious_count += 1


# Count for Serious and serious
import re

serious_count = 0
for row in posts:
    # how many posts have either "[Serious]" or "[serious]" in their title.
    if re.search("\[[Ss]erious\]", row[0]) != None:
        serious_count += 1


# more inconsistency
import re

serious_count = 0
for row in posts:
    # some users have tagged their post using "(Serious)" or "(serious)". Escaping "[", "]", "(", and ")" with the backslash
    if re.search("[\[\(][Ss]erious[\]\)]", row[0]) != None:
        serious_count += 1



"""we can combine our regular expression for the serious tag at the beginning and end with the "|" operator
to match all titles with the serious tag at either the beginning or end"""
import re

serious_start_count = 0
serious_end_count = 0
serious_count_final = 0
for row in posts:
    # Use the "^" character to count how many posts have the serious tag at the beginning of their title.
    if re.search("^[\[\(][Ss]erious[\]\)]", row[0]) != None:
        serious_start_count += 1

for row in posts:
    # Use the "$" character to count how many posts have the serious tag at the end of their title.
    if re.search("[\[\(][Ss]erious[\]\)]$", row[0]) != None:
        serious_end_count += 1

for row in posts:
    # Use the "|" character to count how many posts have the serious tag at either the beginning or end of their title.
    if re.search("^[\[\(][Ss]erious[\]\)]|[\[\(][Ss]erious[\]\)]$", row[0]) != None:
        serious_count_final += 1





"""The re module provides a sub() method that takes the following parameters (in the following order):
1. pattern - The regular expression to match
2. repl    - The string to replace the matched substring with
3. string  - The string in which we would like to replace occurrences of the pattern with repl
"""
"""So, if we were to call re.sub("yo", "hello", "yo world"), the function will replace "yo" with "hello" in "yo world",
producing the result "hello world".
The sub() function simply returns the original string if pattern is not found."""

# Replace "[serious]", "(Serious)", and "(serious)" with "[Serious]" in the titles of every row in posts.
import re
posts_new = []
for row in posts:
    # repl argument is an ordinary string, not a regular expression, so characters like "[" do not need to be escaped.
    row[0] = re.sub("[\[\(][Ss]erious[\]\)]" , "[Serious]", row[0])
    # Append each formatted row to posts_new
    posts_new.append(row)



import re

year_strings = []
# Loop through strings
for string in strings:
    # determine if each string contains a year between 1000 and 2999
   if re.search("[1-2][0-9][0-9][0-9]", string) != None:
       # Store every string that contains a year in year_strings
        year_strings.append(string)




"""Using curly brackets ("{" and "}"), we can indicate that a pattern should repeat.
If we were matching any 4-digit number, we could repeat the pattern "[0-9]" 4 times by writing "[0-9]{4}"."""

import re

year_strings = []
for string in strings:
    # determine if each string contains a year between 1000 and 2999
   if re.search("[1-2][0-9]{3}", string) != None:
        year_strings.append(string)




"""The re module contains a findall() method. findall() returns a list of substrings that match the provided regular expression.
re.findall("[a-z]", "abc123") would return ["a", "b", "c"], since those are the substrings that match the regular expression"""

# extract years from a string
import re
# Use re.findall() to obtain a list of all years between 1000 and 2999 in the string years_string. and assign the result to years
years = re.findall("[1-2][0-9]{3}", years_string)




