# Board Game Reviews
# predict average_rating using the other columns
import pandas
# Read board_games.csv into a Dataframe called board_games using the Pandas library.
board_games = pandas.read_csv("board_games.csv")
# Print out the first few rows of board_games and look closely at the data.
print(board_games.head())

# Use the dropna Dataframe method with the axis argument set to 0 to
# remove any rows that contain missing values.
board_games = board_games.dropna(axis=0)
# Remove any rows in board_games where users_rated equals 0.
# This will remove any rows that have no reviews.
board_games = board_games[board_games["users_rated"] > 0]
# board_games.to_csv("board_games_output.cvs")

# Create a histogram of the average_rating column using the hist function.
import matplotlib.pyplot as plt
plt.hist(board_games["average_rating"])
plt.show()

# Calculate the standard deviation of the average_rating column and print it out.
import numpy
print(numpy.std(board_games["average_rating"]))

# Calculate the mean of the average_rating column and print it out.
print(numpy.mean(board_games["average_rating"]))

# Initialize the KMeans class with 5 clusters
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5)

# Extract the numeric columns of board_games, and assign to the variable numeric_columns.
cols = list(board_games.columns)
cols.remove("name")
cols.remove("id")
cols.remove("type")
numeric_columns = board_games[cols]
# Another way: numeric_columns = board_games.iloc[:, 4:]

# Fit the KMeans class to numeric_columns using the fit method
result = kmeans_model.fit_transform(numeric_columns)
# print(result)
# fit_transform() = fit() + return distance data (ND array of ND array)

# result2 = kmeans_model.fit(numeric_columns)
# print(result2)

game_mean = numeric_columns.apply(numpy.mean, axis=1)
game_std = numeric_columns.apply(numpy.std, axis=1)

# kmeans_model changes itself. no need to assign to itself.
labels = kmeans_model.labels_
plt.scatter(x=game_mean, y=game_std, c=labels)
plt.show()
