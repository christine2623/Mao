# Using the OAuth token 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk

# The User-Agent header will tell Reddit that Dataquest is accessing the API
from pip._vendor import requests

headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}

# Retrieve the top articles from the past day in the /r/python subreddit
# set the parameter t with the value set to the string day to only get the top articles for the past hour.
parameter = {"t": "day"}
# Make a get request and provide the headers and the parameters
response = requests.get("https://oauth.reddit.com/r/python/top", headers=headers, params=parameter)
# get the JSON response data
python_top = response.json()
print(python_top)


# Getting The Most Upvoted Article
# Extract the list containing all of the articles, and assign it to python_top_articles
# To get the articles, you need to get to the data key, then children key, then you can find the article (see documentation)
python_top_articles = python_top["data"]["children"]
most_upvoted = ""
most_upvoted_count = 0
# Iterate through the list to get the top voted article
for article in python_top_articles:
    # You'll have to get into the data key to get the detail info of the article (see documentation)
    art = article["data"]
    # Get the most_upvoted
    # ups is the key and the value is the upvoted times
    if art["ups"]>most_upvoted_count:
        most_upvoted_count = art["ups"]
        most_upvoted = art["id"]

print(most_upvoted)


# Getting Article Comments
# Generate the full URL to query based on the subreddit name and article id.
response = requests.get("https://oauth.reddit.com/r/python/comments/4b7w9u", headers=headers)
comments = response.json()


# Getting The Most Upvoted Comment
# see documentation to know how they store the comments_list
# adding [1] is because the first item in the list is information about the article, and the second item is information about the comments.
comments_list = comments[1]["data"]["children"]
comments_count = 0
most_upvoted_comment = ""
for comment in comments_list:
    com = comment["data"]
    if com["ups"] > comments_count:
        comments_count = com["ups"]
        most_upvoted_comment = com["id"]



# Upvoting A Comment
"""
upvote a comment with the /api/vote endpoint. You'll need to pass in the following parameters:
dir -- vote direction, 1, 0, or -1. 1 is an upvote, and -1 is a downvote.
id -- the id of the article or comment to upvote.
"""

parameter = {"dir": 1, "id": "d16y4ry"}
response = requests.post("https://oauth.reddit.com/api/vote", headers=headers, json=parameter)
status = response.status_code



