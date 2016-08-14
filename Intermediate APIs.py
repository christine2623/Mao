"""
APIs prevent you from making too many requests in too short of a time.
This is known as rate limiting, and ensures that one user cannot overload the API server by making too many requests too fast.
"""



# In this mission, we'll be exploring the Github API and using it to pull some interesting data on repositories and users.
# Github is a site for hosting code.


from pip._vendor import requests

# Create a dictionary of headers, with our Authorization header.
# Get token: https://github.com/settings/tokens
headers = {"Authorization": "token 4570892447dab8aa056479683fbf635c4bdd59c4"}

# Make a GET request to the Github API with our headers.
# This API endpoint will give us details about Vik Paruchuri.
response = requests.get("https://api.github.com/users/VikParuchuri", headers=headers)

# Print the content of the response.  As you can see, this token is associated with the account of Vik Paruchuri.
print(response.json())

# this APT endpoint will tell you which organizations a Github user is in.
response = requests.get("https://api.github.com/users/VikParuchuri/orgs", headers=headers)
# Print the content of the response.
orgs = response.json()
print(orgs)

# Use the endpoint https://api.github.com/users/torvalds and the same headers we used earlier to get information about Linus Torvalds.
response = requests.get("https://api.github.com/users/torvalds", headers=headers)
#  get the json of the response.
torvalds = response.json()
print(torvalds)


# Make a GET request to the https://api.github.com/repos/octocat/Hello-World endpoint.
response = requests.get("https://api.github.com/repos/octocat/Hello-World", headers=headers)
#  get the json of the response.
hello_world = response.json()
print(hello_world)


"""
pagination. This means that the API provider will only return a certain number of records per page.
You can specify the page number that you want to access. To access all of the pages, you'll need to write a loop.
Two parameters for pagination:
page is the page that we want to access,
and per_page is the number of records we want to see on each page.
"""

# Page 1
params = {"per_page": 50, "page": 1}
# get the repositories that a user has starred, or marked as interesting
response = requests.get("https://api.github.com/users/VikParuchuri/starred", headers=headers, params=params)
page1_repos = response.json()

# Page 2
params = {"per_page": 50, "page": 2}
response = requests.get("https://api.github.com/users/VikParuchuri/starred", headers=headers, params=params)
page2_repos = response.json()


# Making a GET request to https://api.github.com/user will give you information about the user that the authentication token is for.
response = requests.get("https://api.github.com/user", headers=headers)
# Assign the decoded json of the result to the user variable
user = response.json()
# response is Response data type
# response.text is str data type
print(user) # dict data type


# GET requests are used to retrieve information from the server (hence the name GET)
# POST requests are used to send information to the server, and create objects on the server.
"""
Two parameters for POST:
name -- required, the name of the repository
description -- optional, the description of the repository
"""
# Not all endpoints will accept a POST request, and not all will accept a GET request.


# Create the data we'll pass into the API endpoint.  This endpoint only requires the "name" key, but others are optional.
payload = {"name": "test"}
# We need to pass in our authentication headers!
response = requests.post("https://api.github.com/user/repos", json=payload, headers=headers)
print(response.status_code)

# Create a new repository named learning-about-apis.
new_repo = {"name": "learning-about-apis"}
# Use the post function to create a new repository at the repository endpoint.
response = requests.post("https://api.github.com/user/repos", json=new_repo, headers=headers)
# Assign the status code a variable status
status = response.status_code
# check if the status code is 201. (completed)
print(status)



# Update an existing object: PATCH and PUT
"""
We use PATCH requests when we want to change a few attributes of an object,
and we don't want to send the whole object to the server (maybe we just want to change the name of our repository, for example).
A PATCH request will usually return a 200 status code if everything goes fine.
"""
"""
We use PUT requests when we want to send the whole object to the server,
and replace the version on the server with the version we went."""


payload = {"description": "The best repository ever!", "name": "test"}
response = requests.patch("https://api.github.com/repos/VikParuchuri/test", json=payload, headers=headers)
print(response.status_code)

# Create a dictionary with the new description you want to patch and the name of the repository
new_description = {"description": "Learning about requests!", "name": "learning-about-apis"}
# Make a PATCH request to the https://api.github.com/repos/VikParuchuri/learning-about-apis endpoint
# that changes the description to Learning about requests!.
response = requests.patch("https://api.github.com/repos/VikParuchuri/learning-about-apis", json=new_description, headers=headers)
status=response.status_code
print(status)


# We can use the DELETE request to remove repositories.
# A successful DELETE request will usually return a 204 request, indicating that the object has been deleted.
response = requests.delete("https://api.github.com/repos/VikParuchuri/test", headers=headers)
print(response.status_code)

# Delete the learning about apis repository
respone = requests.delete("https://api.github.com/repos/VikParuchuri/learning-about-apis", headers=headers)
status=response.status_code
print(status)



# GO revoke the token


