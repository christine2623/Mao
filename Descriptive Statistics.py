import pandas as pd
import matplotlib.pyplot as plt


movie_reviews = pd.read_csv("fandango_score_comparison.csv")
# Create a Matplotlib subplot grid with the following properties, figsize of 4 (width) by 8 (height)
fig = plt.figure(figsize=(5, 12))
# 4 rows by 1 column
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

# each Axes instance should have an x-value range of 0.0 to 5.0
ax1.set_xlim(0,5.0)
ax2.set_xlim(0,5.0)
ax3.set_xlim(0,5.0)
ax4.set_xlim(0,5.0)

# First plot (top most): Histogram of normalized Rotten Tomatoes scores by users.
movie_reviews["RT_user_norm"].hist(ax=ax1)
# Second plot: Histogram of normalized Metacritic scores by users.
movie_reviews["Metacritic_user_nom"].hist(ax=ax2)
# Third plot: Histogram of Fandango scores by users.
movie_reviews["Fandango_Ratingvalue"].hist(ax=ax3)
# Fourth plot (bottom most): Histogram of IMDB scores by users.
movie_reviews["IMDB_norm"].hist(ax=ax4)
# ax4.hist(movie_reviews["IMDB_norm"]),  works too
plt.show()



# Mean
# Write a function, named calc_mean, that returns the mean for the values in a Series object.
# You can return a value in series using the values attribute, however, float object has no attribute "values"

def calc_mean(series):
    vals = series.values
    mean = sum(vals) / len(vals)
    return mean

# Select just the columns containing normalized user reviews and assign to a separate Dataframe named user_reviews.
user_related = ["RT_user_norm", "Metacritic_user_nom", "Fandango_Ratingvalue", "IMDB_norm"]
user_reviews = movie_reviews[user_related]

# Use the Dataframe method apply to apply the calc_mean function over the filtered Dataframe user_reviews.
# dataframe.apply() will call the function on series object while dataframe[col].apply will call the function on  float object
user_reviews_means = user_reviews.apply(calc_mean)


rt_mean = user_reviews_means["RT_user_norm"]
mc_mean = user_reviews_means["Metacritic_user_nom"]
fg_mean = user_reviews_means["Fandango_Ratingvalue"]
id_mean = user_reviews_means["IMDB_norm"]

print("Rotten Tomatoes (mean):", rt_mean)
print("Metacritic (mean):", mc_mean)
print("Fandango (mean):",fg_mean)
print("IMDB (mean):",id_mean)




# Variance And Standard Deviation
# write a function, named calc_variance, that returns the variance for the values in a Series object
# Every element in the series will perform the operation at the same time so there's no need to write a for loop.
def calc_variance(series):
    mean = calc_mean(series)
    squared_deviations = (series - mean)**2
    mean_squared_deviations = calc_mean(squared_deviations)
    return mean_squared_deviations

#  calculate the standard deviation
def calc_std(series):
    std_dev = calc_variance(series) ** (1/2)
    return std_dev

user_reviews_var = user_reviews.apply(calc_variance)
rt_var = user_reviews_var["RT_user_norm"]
mc_var = user_reviews_var["Metacritic_user_nom"]
fg_var = user_reviews_var["Fandango_Ratingvalue"]
id_var = user_reviews_var["IMDB_norm"]

user_reviews_stdev = user_reviews.apply(calc_std)
rt_stdev = user_reviews_stdev["RT_user_norm"]
mc_stdev = user_reviews_stdev["Metacritic_user_nom"]
fg_stdev = user_reviews_stdev["Fandango_Ratingvalue"]
id_stdev = user_reviews_stdev["IMDB_norm"]

print("Rotten Tomatoes (variance):", rt_var)
print("Metacritic (variance):", mc_var)
print("Fandango (variance):", fg_var)
print("IMDB (variance):", id_var)

print("Rotten Tomatoes (standard deviation):", rt_stdev)
print("Metacritic (standard deviation):", mc_stdev)
print("Fandango (standard deviation):", fg_stdev)
print("IMDB (standard deviation):", id_stdev)

"""Findings:
Rotten Tomatoes and Metacritic have more spread out scores (high variance) and the mean is around 3.
Fandango, on the other hand, has low spread (low variance) and a much higher mean,
which could imply that the site has a strong bias towards higher reviews.
IMDB is somewhere in the middle, with a low variance, like Fandango's user reviews, but a much more moderate mean value.

Since Fandango's main business is selling movie tickets,
it's possible their primary incentive may differ from pure review sites like Rotten Tomatoes or Metacritic."""



# Scatter Plots
# if Fandango's user ratings are at least relatively correct.
#  More precisely, are movies that are highly rated on Rotten Tomatoes, IMDB, and Metacritic also highly rated on Fandango?
# Create a Matplotlib subplot grid
fig2 = plt.figure(figsize=(4, 8))
# 3 rows by 1 column
ax1 = fig2.add_subplot(3,1,1)
ax2 = fig2.add_subplot(3,1,2)
ax3 = fig2.add_subplot(3,1,3)

# each Axes instance should have an x-value range of 0.0 to 5.0
ax1.set_xlim(0,5.0)
ax2.set_xlim(0,5.0)
ax3.set_xlim(0,5.0)

# First plot (top most): Fandango user reviews vs. Rotten Tomatoes user reviews.
ax1.scatter(movie_reviews["RT_user_norm"], movie_reviews["Fandango_Ratingvalue"])
# Second plot: Fandango user reviews vs. Metacritic user reviews.
ax2.scatter(movie_reviews["Metacritic_user_nom"], movie_reviews["Fandango_Ratingvalue"])
# Third plot (bottom most): Fandango user reviews vs. IMDB user reviews.
ax3.scatter(movie_reviews["IMDB_norm"], movie_reviews["Fandango_Ratingvalue"])
plt.show()

""" Findings"
It seems like Rotten Tomatoes and IMDB user reviews correlate the most with Fandango user reviews while Metacritic only weakly correlates. """



# Covariance
# Write a function, named calc_covariance, that computes the covariance between the values of 2 Series objects.
def calc_covariance(series1, series2):
    mean1 = calc_mean(series1)
    mean2 = calc_mean(series2)
    difference = (series1-mean1)*(series2-mean2)
    covariance = calc_mean(difference)
    return covariance

# Compute the covariance between the RT_user_norm and Fandango_Ratingvalue columns.
rt_fg_covar = calc_covariance(movie_reviews["RT_user_norm"], movie_reviews["Fandango_Ratingvalue"])
# Compute the covariance between the Metacritic_user_nom and Fandango_Ratingvalue columns.
mc_fg_covar = calc_covariance(movie_reviews["Metacritic_user_nom"], movie_reviews["Fandango_Ratingvalue"])
# Compute the covariance between the IMDB_norm and Fandango_Ratingvalue columns.
id_fg_covar = calc_covariance(movie_reviews["IMDB_norm"], movie_reviews["Fandango_Ratingvalue"])

print("Covariance between Rotten Tomatoes and Fandango:", rt_fg_covar)
print("Covariance between Metacritic and Fandango", mc_fg_covar)
print("Covariance between IMDB and Fandango", id_fg_covar)

"""Findings:
Rotten Tomatoes covaries strongly with Fandango (0.36) compared to Metacritic (0.13) and IMDB (0.14). """


# Correlation
# Write a function, named calc_correlation,
# that uses the calc_covariance and calc_variance functions to calculate the correlation for 2 Series objects.
def calc_correlation(series1, series2):
    cov = calc_covariance(series1, series2)
    std_square = calc_std(series1)*calc_std(series2)
    correlation = cov/std_square
    return correlation

# Compute the correlation between the RT_user_norm and Fandango_Ratingvalue columns and assign the result to rt_fg_corr.
rt_fg_corr = calc_correlation(movie_reviews["RT_user_norm"], movie_reviews["Fandango_Ratingvalue"])
# Compute the correlation between the Metacritic_user_nom and Fandango_Ratingvalue columns and assign the result to mc_fg_corr.
mc_fg_corr = calc_correlation(movie_reviews["Metacritic_user_nom"], movie_reviews["Fandango_Ratingvalue"])
# Compute the correlation between the IMDB_norm and Fandango_Ratingvalue columns and assign the result to id_fg_corr.
id_fg_corr = calc_correlation(movie_reviews["IMDB_norm"], movie_reviews["Fandango_Ratingvalue"])

print("Correlation between Rotten Tomatoes and Fandango", rt_fg_corr)
print("Correlation between Metacritic and Fandango", mc_fg_corr)
print("Correlation between IMDB and Fandango", id_fg_corr)

"""Findings:
As the scatter plots suggested, Rotten Tomatoes and IMDB correlate the strongest with Fandango, with correlation values of 0.72 and 0.60 respectively.
Metacritic, on the other hand, only has a correlation value of 0.34 with Fandango.
While covariance and correlation values may seem complicated to compute and hard to reason with,
their best use case is in comparing relationships like we did in this challenge."""