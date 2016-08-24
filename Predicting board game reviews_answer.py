import pandas

board_games = pandas.read_csv("board_games.csv")
print(board_games.head())
board_games.dropna(axis=0)
board_games = board_games[board_games["users_rated"] > 0]

board_games.head()

import matplotlib.pyplot as plt

plt.hist(board_games["average_rating"])

print(board_games["average_rating"].std())
print(board_games["average_rating"].mean())

from sklearn.cluster import KMeans

clus = KMeans(n_clusters=5)
cols = list(board_games.columns)
cols.remove("name")
cols.remove("id")
cols.remove("type")
numeric = board_games[cols]

clus.fit(numeric)
