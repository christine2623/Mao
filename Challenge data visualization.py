import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
# pd.tools.plotting.scatter_matrix

# how to install scipy and seaborn
# Source: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
# Command: pip install [Local File Location][Your specific file such as scipy-0.16.0-cp27-none-win_amd64.whl]

# Enable inline display of Matplotlib plots.
# %matplotlib inline

hollywood_movies = pd.read_csv("hollywood_movies.csv")
print(hollywood_movies.head(5))
# Select the exclude column and display its distribution using the value_counts method.
print(hollywood_movies["exclude"].value_counts())
# Remove the exclude column, since it only contains null values, using the Dataframe method drop.
hollywood_movies = hollywood_movies.drop("exclude", axis=1)





# Create a new Figure, setting figsize to 6 by 10
fig = plt.figure(figsize=(6,10))
# Create 2 vertically oriented subplots, with one subplot on the top and one subplot on the bottom
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
""" On the top subplot, generate a scatter plot with:
The Profitability column on the x-axis.
The Audience Rating column on the y-axis.
Set the x-axis and y-axis labels to match the column names."""
ax1.scatter(hollywood_movies["Profitability"], hollywood_movies["Audience Rating"])
# Set the title of both subplots to Hollywood Movies, 2007-2011
ax1.set(xlabel = "Profitability", ylabel = "Audience Rating", title = "Hollywood Movies, 2007-2011")
"""On the bottom subplot, generate a scatter plot with:
The Audience Rating column on the x-axis.
The Profitability column on the y-axis.
Set the x-axis and y-axis labels to match the column names."""
ax2.scatter(hollywood_movies["Audience Rating"], hollywood_movies["Profitability"])
ax2.set(xlabel = "Audience Rating", ylabel = "Profitability", title = "Hollywood Movies, 2007-2011")
plt.show()






# Filter out the movie Paranormal Activity from hollywood_movies and assign the resulting Dataframe to normal_movies
normal_movies = hollywood_movies[hollywood_movies["Film"]!="Paranormal Activity"]
# Generate a scatter matrix of the Profitability and Audience Rating columns from normal_movies.
scatter_matrix(normal_movies[["Profitability","Audience Rating"]], figsize = (6,6))
plt.show()




# Use the Pandas Dataframe method plot to generate boxplots for the Critic Rating and Audience Rating columns from normal_movies.
normal_movies[["Critic Rating", "Audience Rating"]].plot(kind = "box")
plt.show()




# Create a Figure instance and set the figsize parameter to (8,4)
fig2 = plt.figure(figsize = (8,4))
# Create 2 horizontally oriented subplots, with one subplot on the left and one subplot on the right.
ax1 = fig2.add_subplot(1,2,1)
ax2 = fig2.add_subplot(1,2,2)
# Use the Dataframe method sort_values to sort normal_movies by the Year column.
normal_movies = normal_movies.sort_values("Year")
# On the left subplot, generate a separate box plot for the values in the Critic Rating column that correspond to each year (from 2007 to 2011).
sns.boxplot(x="Year", y="Critic Rating", data=normal_movies, ax=ax1)
# Use the ax parameter to specify the Axes instance you want this plot to go on
sns.boxplot(x="Year", y="Audience Rating", data=normal_movies, ax=ax2)
plt.show()





# Define a function to seperate profitability into true and false two categories
def is_profitable(row):
    # if profitability <= 1.0, return false
    if row["Profitability"] <= 1.0:
        return False
    # else, return true
    return True
# Use .apply() function to iterate function through the dataframe, axis=1 means iterate through rows
# And create a column named "Profitable" back in the normal_movies dataframe
normal_movies["Profitable"] = normal_movies.apply(is_profitable, axis=1)
print(normal_movies["Profitable"].value_counts())

# Create a Figure instance and set the figsize to (12,6).
fig=plt.figure(figsize=(12,6))
# Add 2 horizontally oriented subplots, with one subplot on the left and one subplot on the right.
ax1=fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
# Use the Seaborn function boxplot to generate a box-and-whisker diagram for the values in Audience Rating for the unprofitable films first
# then a box-and-whisker diagram for the profitable films.
sns.boxplot(x="Profitable", y="Audience Rating", data = normal_movies, ax=ax1)
sns.boxplot(x="Profitable", y="Critic Rating", data = normal_movies, ax=ax2)
plt.show()
