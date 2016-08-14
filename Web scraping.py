# One way to access this data without waiting for the provider to create an API is to use a technique called web scraping.
"""
Web scraping allows us to load a webpage into Python, and extract the information we want.
We can then work with the data using normal data analysis tools, like pandas and numpy.
we'll be heavily using the requests library, which enables us to download a webpage,
and the beautifulsoup library, which enables us to extract the relevant parts of a webpage.
"""

# Webpages are coded in HyperText Markup Language (HTML)
# HTML is not a programming language, like Python, but it is a markup language, and has its own syntax and rules.
from pip._vendor import requests

"""
<p><b> Hello, This is a small world. </b></p>
the b tag is nested within the p tag."""
"""
The composition of an HTML document is broadly split up into the head section,
which contains information that is useful to the web browser that is rendering the page, but isn't shown to the user,
and the body section, which contains the main information that is shown to the user on the page.
"""
# Make a get requests
response = requests.get("http://dataquestio.github.io/web-scraping-pages/simple.html")
# get the content of the response
content = response.content


# Retrieving Elements From A Page
"""
In order to parse the webpage with python, we'll use the BeautifulSoup library.
This library allows us to extract tags from an HTML document."""

from bs4 import BeautifulSoup

# Initialize the parser, and pass in the content we grabbed earlier.
parser = BeautifulSoup(content, 'html.parser')

# Get the body tag from the document.
# Since we passed in the top level of the document to the parser, we need to pick a branch off of the root.
# With beautifulsoup, we can access branches by simply using tag types as
body = parser.body

# Get the p tag from the body.
p = body.p

# Print the text of the p tag.
# Text is a property that gets the inside text of a tag.
print(p.text)

# Initiate a parser
parser = BeautifulSoup(content, "html.parser")
# Get the head tag from the web page
head = parser.head
# Get the title from the head
title = head.title
print(title.text)


# Using Find All
# find_all will find all occurences of a tag in the current element. find_all will return a list.

parser = BeautifulSoup(content, 'html.parser')
# Get a list of all occurences of the body tag in the element.
body = parser.find_all("body")
# Get the paragraph tag
p = body[0].find_all("p")
# Get the text
print(p[0].text)

# Use the find_all method to get the text inside the title tag and assign the result to title_text
parser = BeautifulSoup(content, 'html.parser')
# Find all the "head", it will return a list
head = parser.find_all("head")
# Take the first element of the head list and find all "title"
title = head[0].find_all("title")
# Assign title_text to the text of the first title in the title list
title_text = title[0].text
print(title_text)


# The div tag is used to indicate a division of the page -- it's used to split up the page into logical units.
# We can use find_all to do this, but we'll pass in the additional id attribute.

# Get the page content and setup a new parser.
response = requests.get("http://dataquestio.github.io/web-scraping-pages/simple_ids.html")
content = response.content
parser = BeautifulSoup(content, 'html.parser')

# Pass in the id attribute to only get elements with a certain id.
first_paragraph = parser.find_all("p", id="first")[0]
print(first_paragraph.text)

# find all "p" with id = second and return the first element assign it to second_paragraph
second_paragraph = parser.find_all("p", id="second")[0]
# get the text of the second_paragraph
second_paragraph_text = second_paragraph.text
print(second_paragraph.text)


# Element Classes
# HTML also enables elements to have classes.
"""
All elements with the same class usually share some kind of characteristic that leads the author of the page to group them into the same class.
One element can have multiple classes."""

# Get the website that contains classes.
response = requests.get("http://dataquestio.github.io/web-scraping-pages/simple_classes.html")
# Get the content of the website
content = response.content
parser = BeautifulSoup(content, 'html.parser')

# Get the first inner paragraph.
# Find all the paragraph tags with the class inner-text.
# Then take the first element in that list.
first_inner_paragraph = parser.find_all("p", class_="inner-text")[0]
print(first_inner_paragraph.text)

# Find all the paragraph tags with the class inner_text and get the second element in the list.
second_inner_paragraph = parser.find_all("p", class_="inner-text")[1]
# Extract the text
second_inner_paragraph_text = second_inner_paragraph.text
print(second_inner_paragraph_text)

first_outer_paragraph = parser.find_all("p", class_="outer-text")[0]
first_outer_paragraph_text = first_outer_paragraph.text
print(first_outer_paragraph)


# Cascading Style Sheets, or CSS, is a way to add style to HTML pages.
# CSS uses selectors that select elements and classes of elements to decide where to add certain stylistic touches,
# like color and font size changes.
"""
CSS select classes with . symbol like
p.inner-text{color: red}
select ids with # symbol like
p#first{color: red}

You can also style ids and classes without any tags. This css will make any element with the id first red:
#first{color: red}
And this will make any element with the class inner-text red:
.inner-text{color: red}
"""

# Using CSS Selectors
# With BeautifulSoup, we can use CSS selectors very easily. We just use the .select method.

# Get the website that contains classes and ids
response = requests.get("http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html")
content = response.content
parser = BeautifulSoup(content, 'html.parser')

# Select all the elements with the first-item class
first_items = parser.select(".first-item")

# Print the text of the first paragraph (first element with the first-item class)
print(first_items[0].text)

# select all the elements in outer-text class
outer_text = parser.select(".outer-text")
# get the first element in the list
first_outer_text = outer_text[0].text

# select all the elements in the second ids
second_ids = parser.select("#second")
# get the first element in the list
second_text = second_ids[0].text



# Nesting CSS Selectors
"""
This CSS Selector will select any items with the id first inside a div tag inside a body tag:
body div #first
"""



# Using Nested CSS Selectors

# Get the super bowl box score data.
response = requests.get("http://dataquestio.github.io/web-scraping-pages/2014_super_bowl.html")
content = response.content
parser = BeautifulSoup(content, 'html.parser')

# Find the number of turnovers committed by the Seahawks.
turnovers = parser.select("#turnovers")[0]
seahawks_turnovers = turnovers.select("td")[1]
seahawks_turnovers_count = seahawks_turnovers.text
print(seahawks_turnovers_count)

# Find the items with the total-plays id and return the first in the list
total_plays = parser.select("#total-plays")[0]
# Find the td and return the third item
patriots_total_plays = total_plays.select("td")[2]
# print out the text
patriots_total_plays_count = patriots_total_plays.text


total_yards = parser.select("#total-yards")[0]
seahawks_total_yards = total_yards.select("td")[1]
seahawks_total_yards_count = seahawks_total_yards.text
