# Clustering Overview
"""
So far, we've looked at regression and classification. These are both types of supervised machine learning.
In supervised learning, you train an algorithm to predict an unknown variable from known variables.
"""
"""
Another major type of machine learning is called unsupervised learning. In unsupervised learning,
we aren't trying to predict anything. Instead, we're finding patterns in data.
"""
"""
One of the main unsupervised learning techniques is called clustering.
We use clustering when we're trying to explore a dataset, and understand the connections between the various rows and columns.
"""
"""
Clustering algorithms group similar rows together. There can be one or more groups in the data, and these groups form the clusters.
As we look at the clusters, we can start to better understand the structure of the data.
Clustering is a key way to explore unknown data, and it's a very commonly used machine learning technique."""





# The Dataset
"""
Clustering voting data of Senators is particularly interesting because it can expose patterns that go deeper than party affiliation.
For example, some Republicans are more liberal than the rest of their party.
Looking at voting data can help us discover the Senators who are more or less in the mainstream of their party.
"""
import pandas

votes = pandas.read_csv("114_congress.csv")




# Exploring The Data
# Find how many Senators are in each party.
# Use the value_counts() method on the party column of votes. Print the results.
num_senators = votes["party"].value_counts()
print(num_senators)
# Find what the "average" vote for each bill was.
# Use the mean() method on the votes Dataframe.
mean_votes = votes.mean()
print(mean_votes)
# If the mean for a column is less than .5, more Senators voted against the bill, and vice versa if it's over .5.





# Distance Between Senators
"""
To group Senators together, we need some way to figure out how "close" the Senators are to each other.
We'll then group together the Senators that are the closest.
We can actually discover this distance mathematically, by finding how similar the votes of two Senators are.
The closer together the voting records of two Senators, the more ideologically similar they are
(voting the same way indicates that you share the same views).
"""
# To find the distance between two rows, we can use Euclidean distance. d=((q1−p1)**2+(q2−p2)**2+...+(qn−pn)**2)**(1/2)
# we can use the euclidean_distances() method in the scikit-learn library.
# euclidean_distances(votes.iloc[0,3:], votes.iloc[1,3:])
from sklearn.metrics.pairwise import euclidean_distances
# A numpy matrix can be reshaped into a vector using reshape function with parameter -1
print(euclidean_distances(votes.iloc[0,3:].reshape(1, -1), votes.iloc[1,3:].reshape(1, -1)))

# Compute the Euclidean distance between the first row and the third row.
distance = euclidean_distances(votes.iloc[0,3:].reshape(1, -1), votes.iloc[2, 3:].reshape(1, -1))
print(distance)




# Initial Clustering
"""
We'll use an algorithm called k-means clustering to split our data into clusters.
k-means clustering uses Euclidean distance to form clusters of similar Senators.
it's important to understand clustering at a high level, so we'll leverage the scikit-learn library to train a k-means model.
"""
"""
The k-means algorithm will group Senators who vote similarly on bills together, in clusters.
Each cluster is assigned a center, and the Euclidean distance from each Senator to the center is computed.
Senators are assigned to clusters based on which one they are closest to.
From our background knowledge, we think that Senators will cluster along party lines.
"""
"""
The k-means algorithm requires us to specify the number of clusters upfront.
Because we suspect that clusters will occur along party lines,
and the vast majority of Senators are either Republicans or Democrats, we'll pick 2 for our number of clusters.
"""
"""
We'll use the KMeans class from scikit-learn to perform the clustering.
Because we aren't predicting anything, there's no risk of overfitting, so we'll train our model on the whole dataset.
After training, we'll be able to extract cluster labels that indicate what cluster each Senator belongs to.
kmeans_model = KMeans(n_clusters=2, random_state=1)
a random state of 1 to allow for the same results to be reproduced whenever the algorithm is run.
"""
# We'll then be able to use the fit_transform() method to fit the model to votes and get the distance of each Senator to each cluster.

import pandas as pd
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=1)
# Use the fit_transform() method to fit kmeans_model on the votes DataFrame. Only select columns after the first 3 from votes when fitting.
senator_distances = kmeans_model.fit_transform(votes.iloc[:, 3:])
"""
The result of the fit_transform() is a NumPy array with two columns.
The first column is the Euclidean distance from each Senator to the first cluster,
and the second column is the Euclidean distance to the the second cluster.
The values in the columns will indicate how "far" the Senator is from each cluster.
The further away from the cluster, the less the Senator's voting history aligns with the voting history of the cluster.
"""
print(senator_distances)





# Exploring The Clusters
"""
We can use the Pandas method crosstab() to compute and display how many Senators from each party ended up in each cluster.
The crosstab() method takes in two vectors or Pandas Series and computes
how many times each unique value in the second vector occurs for each unique value in the first vector.
"""
"""
We can extract the cluster labels for each Senator from kmeans_model using kmeans_model.labels_,
then we can make a table comparing these labels to votes["party"] with crosstab().
This will show us if the clusters tend to break down along party lines or not.
"""
# Use the labels_ attribute to extract the labels from kmeans_model. Assign the result to the variable labels.
labels = kmeans_model.labels_
# Use the crosstab() method to print out a table comparing labels to votes["party"], in that order.
# crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
print(pandas.crosstab(labels, votes["party"], rownames=['labels'], colnames=['party']))




# Exploring Senators In The Wrong Cluster
"""
3 Democrats are more similar to Republicans in their voting than their own party.
Let's explore these 3 in more depth so we can figure out why that is.
"""
# We can do this by subsetting votes to only select rows where the party column is D, and the labels variable is 1,
# indicating that the Senator is in the second cluster
# When subsetting a DataFrame with multiple conditions, each condition needs to be in parentheses, and separated by &.

# Select all Democrats in votes who were assigned to the first cluster. Assign the subset to democratic_outliers.
democratic_outliers = votes[(votes["party"] == "D") & (labels == 0)]
# Print out democratic_outliers.
print(democratic_outliers)




# Plotting Out The Clusters
#  We can treat these distances as x and y coordinates, and make a scatterplot that shows the position of each Senator.

# Make a scatterplot using plt.scatter(). Pass in the following keyword arguments:
# x should be the first column of senator_distances.
# y should be the second column of senator_distances.
# c should be labels. This will shade the points according to label. (c=color)
import matplotlib.pyplot as plt
plt.scatter(senator_distances[:,0], senator_distances[:,1], c=labels)
# Use plt.show() to show the plot.
plt.show()





# Finding The Most Extreme
# The most extreme Senators are those who are the furthest away from one cluster.
"""
we'll cube the distances in both columns of senator_distances, then add them together.
The higher the exponent we raise a set of numbers to, the more separation we'll see between small values and low values.
"""

# Compute an extremism rating by cubing every value in senator_distances, then finding the sum across each row.
# Assign the result to extremism.
import numpy
cube_distance = senator_distances ** 3
extremism = numpy.sum(cube_distance, axis=1)
# Assign the extremism variable to the extremism column of votes.
votes["extremism"] = extremism
# Sort votes on the extremism column, in descending order, using the sort_values() method on DataFrames.
votes.sort_values("extremism", ascending=False, inplace=True)
# Print the top 10 most extreme Senators.
print(votes.head(10))


"""
Clustering is a powerful way to explore data and find patterns.
Unsupervised learning is very commonly used with large datasets where it isn't obvious how to start with supervised machine learning.
In general, it's a good idea to try unsupervised learning to explore a dataset
before trying to use supervised learning machine learning models.
"""