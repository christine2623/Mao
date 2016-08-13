"""
Here are a few times when a dataset isn't a good solution:
1. The data are changing quickly. An example of this is stock price data.
It doesn't really make sense to regenerate a dataset and download it every minute -- this will take a lot of bandwidth, and be pretty slow.
2. You want a small piece of a much larger set of data. Reddit comments are one example.
What if you want to just pull your own comments on reddit? It doesn't make much sense to download the entire reddit database, then filter just your own comments.
3. There is repeated computation involved. Spotify has an API that can tell you the genre of a piece of music.
 You could theoretically create your own classifier, and use it to categorize music, but you'll never have as much data as Spotify does.
 In cases like the ones above, an Application Program Interface (API) is the right solution.
"""

# APIs are used to dynamically query and retrieve data.
# APIs work much the same way, except instead of your web browser asking for a webpage,
# your program asks for data. This data is usually returned in json format.

# There are many different types of requests. The most commonly used one, a GET request, is used to retrieve data.

# Import requests library
# Make a get request to get the latest position of the international space station from the opennotify api.
from pip._vendor import requests

response = requests.get("http://api.open-notify.org/iss-now.json")
status_code = response.status_code

"""
The request we just made had a status code of 200. Status codes are returned with every request that is made to a web server.
Status codes indicate information about what happened with a request. Here are some codes that are relevant to GET requests:

200 -- everything went okay, and the result has been returned (if any)
301 -- the server is redirecting you to a different endpoint.
    This can happen when a company switches domain names, or an endpoint name is changed.
401 -- the server thinks you're not authenticated. This happens when you don't send the right credentials to access an API
    (we'll talk about this in a later mission).
400 -- the server thinks you made a bad request. This can happen when you don't send along the right data, among other things.
403 -- the resource you're trying to access is forbidden -- you don't have the right permissions to see it.
404 -- the resource you tried to access wasn't found on the server.
"""

response2 = requests.get("http://api.open-notify.org/iss-pass")
status_code = response2.status_code
# iss-pass wasn't a valid endpoint, so we got a 404 status code in response. We forgot to add .json at the end

response2 = requests.get("http://api.open-notify.org/iss-pass.json")
status_code = response2.status_code
# we got a 400 status code, which indicates a bad request.
# look at the documentation for the OpenNotify API, you'll see that the ISS Pass endpoint requires two parameters.
"""
We can do this by adding an optional keyword argument, params, to our request. In this case,
there are two parameters we need to pass:
lat -- The latitude of the location we want.
lon -- The longitude of the location we want.
We can make a dictionary with these parameters, and then pass them into the function.
"""

# Set up the parameters we want to pass to the API.
# This is the latitude and longitude of San Francisco.
parameters = {"lat": 37.78, "lon": -122.41}
# Make a get request with the parameters.
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
# Retrieve the content of the response
content = response.content
# And print it
print(content)
# This gets the same data as the command above
response = requests.get("http://api.open-notify.org/iss-pass.json?lat=37.78&lon=-122.41")
print(response.content)

# JavaScript Object Notation (JSON). JSON is a way to encode data structures like lists and dictionaries to strings
# that ensures that they are easily readable by machines.
# JSON is the primary format in which data is passed back and forth to APIs.

"""
The json library has two main methods:
dumps -- Takes in a Python object, and converts it to a string.
loads -- Takes a json string, and converts it to a Python object.
"""

# Make a list of fast food chains.
best_food_chains = ["Taco Bell", "Shake Shack", "Chipotle"]
print(type(best_food_chains))

# Import the json library
import json

# Use json.dumps to convert best_food_chains to a string.
best_food_chains_string = json.dumps(best_food_chains)
print(type(best_food_chains_string))

# Convert best_food_chains_string back into a list
print(type(json.loads(best_food_chains_string)))

# Make a dictionary
fast_food_franchise = {
    "Subway": 24722,
    "McDonalds": 14098,
    "Starbucks": 10821,
    "Pizza Hut": 7600
}

# We can also dump a dictionary to a string and load it.
fast_food_franchise_string = json.dumps(fast_food_franchise)
print(type(fast_food_franchise_string))

# Convert fast_food_franchise_string back into a dictionary
fast_food_franchise_2 = json.loads(fast_food_franchise_string)
print(type(fast_food_franchise_2))



# Make the same request we did 2 screens ago.
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

# Get the response data as a python object.  Verify that it's a dictionary.
data = response.json()
print(type(data))
print(data)

# Get the duration value of the first pass of the ISS over San Francisco
# (this is the duration key of the first dictionary in the response list).
first_pass_duration = data["response"][0]["duration"]

# The metadata containing information on how the data was generated and how to decode it is stored in the response headers.
# Headers is a dictionary
print(response.headers)
# Get content-type from response.headers
content_type = response.headers["content-type"]

# Call the API on astros endpoint
data = requests.get("http://api.open-notify.org/astros.json")
# Change the datatype to Python object(dictionary)
in_space_dict = data.json()
# Get the number of how many people are currently in space
in_space_count = in_space_dict["number"]
print(in_space_count)
