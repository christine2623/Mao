# Clustering NBA Players
import pandas as pd
import numpy as np

nba = pd.read_csv("nba_stats.csv")
print(nba.head(3))



# Point Guards
# Point guards play one of the most crucial roles on a team because their primary responsibility is to create scoring opportunities for the team.
"""
machine learning technique called clustering, which allows us to visualize the types of point guards as well as group similar point guards together.
"""
"""
For point guards, it's widely accepted that the Assist to Turnover Ratio is a good indicator
for performance in games as it quantifies the number of scoring opportunities that player created.
Let's also use Points Per Game, since effective Point Guards not only set up scoring opportunities but also take a lot of the shots themselves.
"""
# Create a new Dataframe which contains just the point guards from the data set.
# Point guards are specified as PG in the pos column.
# Assign the filtered data frame to point_guards.
point_guards = nba[nba["pos"] == "PG"]
print(point_guards.head())




# Points Per Game
# Calculate the Points Per Game values for each PG
point_guards["ppg"] = point_guards["pts"] / point_guards["g"]
print(point_guards.head())




# Assist Turnover Ratio
# create a column, atr, for the Assist Turnover Ratio, which is calculated by dividing total assists (ast) by total turnovers (tov)

# Drop the players who have 0 turnovers. Not only did these players only play a few games, making it hard to understand their true abilities,
# but we also cannot divide by 0 when we calculate atr.
point_guards = point_guards[point_guards["tov"] != 0]
point_guards["atr"] = point_guards["ast"] / point_guards["tov"]
print(point_guards.head())




# Visualizing The Point Guards
# Use matplotlib to create a scatter plot with Points Per Game (ppg) on the X axis and Assist Turnover Ratio (atr) on the Y axis.
import matplotlib.pyplot as plt
plt.scatter(point_guards["ppg"], point_guards["atr"])
plt.title("Point Guards")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()



# Clustering Players
# We can use a technique called clustering to segment all of the point guards into groups of alike players.
"""
While regression and other supervised machine learning techniques work well
when we have a clear metric we want to optimize for and lots of pre-labelled data,
we need to instead use unsupervised machine learning techniques to explore the structure within a data set
that doesn't have a clear value to optimize.
"""
"""
Centroid based clustering works well when the clusters resemble circles with centers (or centroids).
The centroid represent the arithmetic mean of all of the data points in that cluster.
"""
"""
K-Means Clustering is a popular centroid-based clustering algorithm that we will use.
The K in K-Means refers to the number of clusters we want to segment our data into.
The key part with K-Means (and most unsupervised machine learning techniques) is that we have to specify what k is.
There are advantages and disadvantages to this, but one advantage is that we can pick the k that makes the most sense for our use case.
We'll set k to 5 since we want K-Means to segment our data into 5 clusters.
"""




# The Algorithm
"""
Setup K-Means is an iterative algorithm that
switches between recalculating the centroid of each cluster and the players that belong to that cluster.
To start, select 5 players at random and assign their coordinates as the initial centroids of the just created clusters.
"""
"""
Step 1 (Assign Points to Clusters)
For each player, calculate the Euclidean distance between that player's coordinates, or values for atr & ppg,
and each of the centroids' coordinates.
Assign the player to the cluster whose centroid is the closest to, or has the lowest Euclidean distance to, the player's values.
"""
"""
Step 2 (Update New Centroids of the Clusters)
For each cluster, compute the new centroid by calculating the arithmetic mean of all of the points (players) in that cluster.
We calculate the arithmetic mean by taking the average of all of the X values (atr) and the average of all of the Y values (ppg) of the points
in that cluster.
"""
"""
Iterate Repeat steps 1 & 2 until the clusters are no longer moving and have converged.
"""
num_clusters = 5
# Use numpy's random function to generate a list, length: num_clusters, of indices
random_initial_points = np.random.choice(point_guards.index, size=num_clusters)
# Use the random indices to create the centroids
# centroids is a dataframe data type
centroids = point_guards.ix[random_initial_points]




# Visualize Centroids
plt.scatter(point_guards["ppg"], point_guards["atr"], c="red")
plt.scatter(centroids["ppg"], centroids["atr"], c="blue")
plt.title("Centroids")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()




# Setup (Continued)
# let's use a dictionary object instead to represent the centroids.
# key: cluster_id of that centroid's cluster ;  value: centroid's coordinates expressed as a list ( ppg value first, atr value second )

# We'll write a function, centroids_to_dict, that takes in the centroids data frame object,
# creates a cluster_id and converts the ppg and atr values for that centroid into a list of coordinates,
# and adds both the cluster_id and coordinates_list into the dictionary that's returned.
def centroids_to_dict(df):
    dic_of_centroids = {}
    # Another way: dictionary = dict()
    for index in range(0, df.shape[0]):
        cluster_id = index
        list_of_coordinates = []
        # df.iloc[index] !!!!
        list_of_coordinates.append(df.iloc[cluster_id]["ppg"])
        list_of_coordinates.append(df.iloc[cluster_id]["atr"])
        dic_of_centroids[cluster_id] = list_of_coordinates
    return dic_of_centroids
"""Another way of create this function:
def centroids_to_dict(centroids):
    dictionary = dict()
    # iterating counter we use to generate a cluster_id
    counter = 0

    # iterate a pandas data frame row-wise using .iterrows()
    for index, row in centroids.iterrows():
        coordinates = [row['ppg'], row['atr']]
        dictionary[counter] = coordinates
        counter += 1

    return dictionary
"""

centroids_dic = centroids_to_dict(centroids)
print(centroids_dic)




# Step 1 (Euclidean Distance)
"""
Before we can assign players to clusters, we need a way to compare the ppg and atr values of the players with each cluster's centroids.
Euclidean distance is the most common technique used in data science for measuring distance between vectors and works extremely well in 2 and 3 dimensions.
While in higher dimensions, Euclidean distance can be misleading, in 2 dimensions Euclidean distance is essentially the Pythagorean theorem
"""
# Create the function calculate_distance, which takes in 2 lists (the player's values for ppg and atr and the centroid's values for ppg and atr).
def calculate_distance(list1, list2):
    total = []
    for index in range(0, len(list1)):
        difference = (list1[index] - list2[index])**2
        total.append(difference)
    distance = (np.sum(total))**(1/2)
    return distance

q = [5, 2]
p = [3,1]
# Sqrt(5) = ~2.24
print(calculate_distance(q, p))




# Step 1 (Continued)
# Create a function that can be applied to every row in the data set (using the apply function in pandas).

new_centroids = centroids[["ppg", "atr"]]

# For each player, we want to calculate the distances to each cluster's centroid using euclidean_distance.
# Create the function, `assign_to_cluster`
def assign_to_cluster(row):
    euclidean_distance = {}
    euclidean_distance_cluster = 0
    for index in range(0, new_centroids.shape[0]):
        difference = ((row["ppg"] - new_centroids.iloc[index]["ppg"]) ** 2) +((row["atr"] - new_centroids.iloc[index]["atr"]) ** 2)
        distance = difference ** (1 / 2)
        # Once we know the distances, we can determine which centroid is the closest (has the lowest distance) and return that centroid's cluster_id.
        if index == 0:
            euclidean_distance["cluster_id"] = index
            euclidean_distance["value"] = distance
            euclidean_distance_cluster = index
        elif index > 0:
            if distance < euclidean_distance["value"]:
                euclidean_distance["cluster_id"] = index
                euclidean_distance["value"] = distance
                euclidean_distance_cluster = index
    return euclidean_distance_cluster

# Create a new column, cluster, that contains the row-wise results of assign_to_cluster.
point_guards["cluster"] = point_guards.apply(assign_to_cluster, axis=1)
print(point_guards.head())




# Visualizing Clusters
def visualize_clusters(df, num_list):
    colors = ["green", "orange", "blue", "yellow", "black"]
    for index in range(0, num_list):
        clustered_df = df[df["cluster"] == index]
        plt.scatter(clustered_df["ppg"], clustered_df["atr"], c=colors[index])
        plt.xlabel('Points Per Game', fontsize=13)
        plt.ylabel('Assist Turnover Ratio', fontsize=13)
    plt.show()

result = visualize_clusters(point_guards, 5)

# range(5) = range(0,5)




# Step 2
# recalculate the centroids for each cluster
# Finish the function, recalculate_centroids, that:
# takes in point_guards,
def recalculate_centroids(df):
    new_centroids_dict = {}
    for index in range(0, num_clusters):
        # uses each cluster_id(from 0 to num_clusters - 1) to pull out all of the players in each cluster
        clustered_df = df[df["cluster"] == index]
        # calculates the new arithmetic mean
        mean_x = np.mean(clustered_df["ppg"])
        mean_y = np.mean(clustered_df["atr"])
        # Another way: new_centroid = [np.average(values_in_cluster['ppg']), np.average(values_in_cluster['atr'])]
        new_centroids_dict[index] = (mean_x, mean_y)
    # and adds the cluster_id and the new arithmetic mean to new_centroids_dict, the final dictionary to be returned.
    return new_centroids_dict

centroids_dict = recalculate_centroids(point_guards)
print(centroids_dict)




# Repeat Step 1
# Now that we recalculated the centroids, let's re-run Step 1 (assign_to_cluster) and see how the clusters shifted.
point_guards["cluster"] = point_guards.apply(assign_to_cluster, axis=1)
result = visualize_clusters(point_guards, 5)




# Repeat Step 2 And Step 1
# Now we need to recalculate the centroids, and shift the clusters again.
centroids_dict = recalculate_centroids(point_guards)
point_guards["cluster"] = point_guards.apply(assign_to_cluster, axis=1)
visualize_clusters(point_guards, num_clusters)




# Challenges Of K-Means
"""
As you repeat Steps 1 and 2 and run visualize_clusters,
you'll notice that a few of the points are changing clusters between every iteration (especially in areas where 2 clusters almost overlap),
but otherwise, the clusters visually look like they don't move a lot after every iteration. This means 2 things:

1. K-Means doesn't cause massive changes in the makeup of clusters between iterations, meaning that it will always converge and become stable
2. Because K-Means is conservative between iterations,
where we pick the initial centroids and how we assign the players to clusters initially matters a lot

To counteract these problems, the sklearn implementation of K-Means does some intelligent things like
re-running the entire clustering process lots of times with random initial centroids
so the final results are a little less biased on one pass-through's initial centroids.
"""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(point_guards[['ppg', 'atr']])
# Use the labels_ attribute to extract the labels from kmeans_model. Assign the result to the variable labels.
point_guards['cluster'] = kmeans.labels_

visualize_clusters(point_guards, num_clusters)


"""Conclusion:
In this lesson, we explored how to segment NBA players into groups with similar traits.
Our exploration helped us get a sense of the 5 types of point guards as based on each player's Assist Turnover Ratio and Points Per Game.
"""