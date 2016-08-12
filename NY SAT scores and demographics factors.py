# investigating the relationship between SAT scores and demographic factors


import pandas

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

# Create a dictionary to quickly return the certain dataframe
data = {}

for dataframe in data_files:
    # '{1} {0}'.format(1, 2) will output "2  1"
    filename = pandas.read_csv("{0}".format(dataframe))
    # string.replace("i","a") output will be "strang"
    key_name = dataframe.replace(".csv", "")
    data[key_name] = filename




print(data["sat_results"].head())



# Explore the dataframes
for key in data:
    print(key, data[key].head())


# Import the .txt file
all_survey = pandas.read_csv("survey_all.txt", delimiter="\t", encoding="windows-1252")
d75_survey = pandas.read_csv("survey_d75.txt", delimiter="\t", encoding="windows-1252")
# concat() combined Dataframe x and Dataframe Y. Dataframe z will have as many rows as the rows in x plus the rows in y.
# z = pandas.concat([x,y], axis=0)
survey = pandas.concat([all_survey, d75_survey], axis=0)
print(survey.head())



# Cleaning data. We only want relevant column in the survey dataframe then add it to the data dictionary
# We can copy column like this: survey["new_column"] = survey["old_column"]
survey["DBN"] =survey["dbn"]
survey_columns = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_10", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]
survey = survey.loc[:, survey_columns]
# Add the key and the dataframe into the data dictionary
data["survey"] = survey



# We want to create a "DBN" column for the dataframe which misses it so we can do the merge later.
# Copy column and rename the column
data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]
# create a function to convert the csd column to the one we want
def paddedcsd(csd):
    str_csd = str(csd)
    if len(str_csd) == 1:
        return "0" + str_csd
    else:
        return str_csd
# apply the function to the CSD column
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(paddedcsd)
# dataframe["new_column"] = dataframe["column_one"] + dataframe["column_two"]
# combine the strings in both columns
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]

print(data["class_size"]["DBN"].head())





# Add up the three SAT score into one total score. It will be easier to plot one score than three.
# use the .to_numeric function to switch the datatype to numeric
# errors="coerce" when we call to_numeric, any invalid strings that can't be converted to numbers are instead treated as missing values.
data["sat_results"]["SAT Math Avg. Score"] = pandas.to_numeric(data["sat_results"]["SAT Math Avg. Score"], errors="coerce")
data["sat_results"]["SAT Writing Avg. Score"] = pandas.to_numeric(data["sat_results"]["SAT Writing Avg. Score"], errors="coerce")
data["sat_results"]["SAT Critical Reading Avg. Score"] = pandas.to_numeric(data["sat_results"]["SAT Critical Reading Avg. Score"], errors="coerce")
# add up all the score and create a new column for it
data["sat_results"]["sat_score"] = data["sat_results"]["SAT Math Avg. Score"] + data["sat_results"]["SAT Writing Avg. Score"] + data["sat_results"]["SAT Critical Reading Avg. Score"]

print(data["sat_results"]["sat_score"].head())





# Get the latitude from the location 1 column
import re

# re.findall("\(.+, .+\)", "1110 Boston Road\nBronx, NY 10456\n(40.8276026690005, -73.90447525699966)")
def find_coordinates(strings):
    coordinates = re.findall("\(.+,.+\)", strings)
    # 'list' object cannot ".split" ; "string" object can do ".split"
    new_coordinates = str(coordinates).split(",")
    new_lat_coordinates = new_coordinates[0]
    # 'list' object cannot ".replace"; "string" object can do ".replace"
    lat = str(new_lat_coordinates).replace("['(","")
    return lat

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_coordinates)
print(data["hs_directory"].head())




# Get the longitude from the location 1 column

def find_long(strings):
    coordinates = re.findall("\(.+,.+\)", strings)
    new_coordinates = coordinates[0].split(",")[1].replace(")","")
    return new_coordinates
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_long)

# Convert the lat and lon to numeric datatype
# specify the errors="coerce" to handle missing values
data["hs_directory"]["lon"] = pandas.to_numeric(data["hs_directory"]["lon"], errors="coerce")
data["hs_directory"]["lat"] = pandas.to_numeric(data["hs_directory"]["lat"], errors="coerce")

print(data["hs_directory"].head())





# condense some of the datasets so that each value in the DBN column is unique
# For class_size dataframe, Filter class_size so the GRADE column only contains the value 09-12
# and the PROGRAM TYPE column only contains the value GEN ED
class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

print(class_size.head())



# Find the average values for each column for each DBN in class_size
# Each school provide different subjects so the DBN is not yet unique in the dataset
# We will combine all the classes provided by the same school and get the avg class size

import numpy
# grouped = obj.groupby(key)
grouped = class_size.groupby("DBN")
# grouped.aggregate(np.sum)
class_size = grouped.aggregate(numpy.mean)
# Above two lines can be written as: class_size = class_size.groupby("DBN").agg(numpy.mean)

"""After grouping a Dataframe and aggregating based on it, the index will become the column the grouping was done on (in this case DBN),
and it will no longer be a column. In order to move DBN back to a column, we'll need to use reset_index. """
# DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
class_size.reset_index(inplace=True)
# Assign class_size back to the class_size key of the data dictionary.
data["class_size"] = class_size

print(data["class_size"].head())




# For demographics dataframe, we only want to select the latest year which is 20112012
data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

print(data["demographics"].head())




# For graduation dataframe, we will choose the data to be the most recent Cohort available, 2006
# We also want data from the full cohort, so we'll only pick rows where Demographic is Total Cohort.

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]

print(data["graduation"].head())






"""
The Advanced Placement, or AP, exams are taken by high school students.
There are several AP exams, each corresponding to a school subject.
If a high schooler passes a test with a high score, they may receive college credit.
The AP is scored on a 1 to 5 scale, with anything 3 or higher being a "passing" score.
Many high schoolers, particularly those who go to academically challenging high schools, take AP exams. """
# see if AP exam scores are correlated with SAT scores across high schools.

# convert the AP exam scores in the ap_2010 dataset to numeric values first.
data["ap_2010"]["AP Test Takers "] = pandas.to_numeric(data["ap_2010"]["AP Test Takers "], errors="coerce")
data["ap_2010"]["Total Exams Taken"] = pandas.to_numeric(data["ap_2010"]["Total Exams Taken"], errors="coerce")
data["ap_2010"]["Number of Exams with scores 3 4 or 5"] = pandas.to_numeric(data["ap_2010"]["Number of Exams with scores 3 4 or 5"], errors="coerce")

print(data["ap_2010"].head())





# this project is figuring out what correlates with SAT score,
# we'll want to preserve as many rows as possible from sat_results while minimizing null values
# assign data["sat_results"] to the variable combined. We'll then merge all the other Dataframes with combined.
# At the end, combined will have all of the columns from all of the datasets.

combined = data["sat_results"]

# perform left join on the merge with the column DBN as how we gonna merge
combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

print(combined.head())
print(combined.shape[0])

# these files contain information that's more valuable to our analysis,
# and because they have fewer missing DBN values, we'll use the inner join type when merging these into combined
combined = combined.merge(data["class_size"], on="DBN")
combined = combined.merge(data["demographics"], on="DBN")
combined = combined.merge(data["survey"], on="DBN")
combined = combined.merge(data["hs_directory"], on="DBN")

print(combined.head())
print(combined.shape[0])





# Now we have a lot of missing data due to left join
# we'll just fill in the missing values with the mean value of the column.
# means = df.mean()
# df = df.fillna(means)
# Then, fill any remaining NaN or null values after the initial replacement with the value 0

means = combined.mean()
combined = combined.fillna(means)
combined = combined.fillna(0)

print(combined.head())






# Then we would like to map out statistics on a school district level.
# It will be useful to add a column that specifies the school district to the dataset.
# We can use indexing to extract the first few characters of a string
# name = "Sinbad"
# print(name[0:2])

def get_school_district(string):
    return string[0:2]


combined["school_dist"] = combined["DBN"].apply(get_school_district)

print(combined["school_dist"].head(30))







# Finding Correlations: using the r value, also called Pearson's correlation coefficient,
# which measures how closely two sequences of numbers are correlated
# An r value falls between -1 and 1, and tells you if the two columns are positively correlated, not correlated, or negatively correlated
"""
In general r-values above .25 or below -.25 are enough to qualify a correlation as interesting.
An r-value isn't perfect, and doesn't indicate that there is a correlation.
It just indicates the possiblity of one.
To really assess whether or not a correlation exists, you need to look at the data using a scatterplot,
and see the "shape" of the data."""

# DataFrame.corr(method='pearson', min_periods=1)
"""
We can use the Pandas corr method to find correlations between columns in a Dataframe.
The result of the method is a Dataframe where each column and row index is the name of a column in the original dataset."""

correlations = combined.corr(method='pearson')
# Filter correlations so that only correlations for the column sat_score are shown.
correlations = correlations["sat_score"]

print(correlations)





# plotting to see if there is any interesting patterns
# df.plot.scatter(x="A", y="b")

# %matplotlib inline
import matplotlib.pyplot as plt
combined.plot.scatter(x="total_enrollment", y="sat_score")


# it doesn't appear that there's an extremely strong correlation between sat_score and total_enrollment
# there's a large cluster of schools, then a few schools going off in 3 different directions.
# there is an interesting cluster of points at the bottom left where total_enrollment and sat_score are both low
# This cluster may be what is causing our r-value to be so high. It's worth extracting the names of the schools in this cluster

# Filter the combined Dataframe, and only keep rows where total_enrollment is under 1000
low_enrollment = combined[combined["total_enrollment"] < 1000]
# sat_score is under 1000
low_enrollment = low_enrollment[low_enrollment["sat_score"] < 1000]

print(low_enrollment["School Name"])


# we found that most of the high schools with low total enrollment and low SAT scores
# are actually schools with a high percentage of English language learners enrolled
# This indicates that it's actually ell_percent that correlates strongly with sat_score instead of total_enrollment
# let's plot out ell_percent vs sat_score

combined.plot.scatter(x="ell_percent", y="sat_score")
plt.show()


# there's still the cluster with very high ell_percent and low sat_score,
# which is the same group of international high schools that we investigated earlier.
# In order to explore this relationship, we'll want to map out ell_percent by school district,
# so we can more easily see which parts of the city have a lot of English language learners.
#  The basemap package allows us to create high-quality maps, plot points over them, then draw coastlines and other features.
# set up the map like this:
"""
m = Basemap(
    projection='merc',
    llcrnrlat=40.496044,
    urcrnrlat=40.915256,
    llcrnrlon=-74.255735,
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.fillcontinents(color='white',lake_color='#85A6D9')"""


from mpl_toolkits.basemap import Basemap

m = Basemap(projection = "merc", llcrnrlat=40.496044,
    urcrnrlat=40.915256,
    llcrnrlon=-74.255735,
    urcrnrlon=-73.700272,
    resolution='i')

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.fillcontinents(color='white',lake_color='#85A6D9')

# Convert from Series objects to List objects.
longitudes = combined["lon"].tolist()
latitudes = combined["lat"].tolist()

# draw latitude and longitude on a scatter plot
# s=20 to increase the size of the points in the scatterplot.
# zorder=2 to plot the points on top of the rest of the map. Otherwise the points will show up underneath the land.
# latlon=True to indicate that we're passing in latitude and longitude coordinates, not axis coordinates.
m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True)

plt.show()




# Now we can start to display meaningful information on maps, such as the percentage of English language learners by area.

from mpl_toolkits.basemap import Basemap

m = Basemap(projection = "merc", llcrnrlat=40.496044,
    urcrnrlat=40.915256,
    llcrnrlon=-74.255735,
    urcrnrlon=-73.700272,
    resolution='i')

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.fillcontinents(color='white',lake_color='#85A6D9')

# Convert from Series objects to List objects.
longitudes = combined["lon"].tolist()
latitudes = combined["lat"].tolist()

# draw latitude and longitude on a scatter plot
# c keyword argument will accept a sequence of numbers, and will shade points corresponding to lower numbers or higher numbers differently.
# c argument shows that we are going to plot the ell_percent column
# cmap stands for colormap. There are plenty of them.
# we'll use the summer colormap, which results in green points when the associated number is low, and yellow when it's high
m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True, c=combined["ell_percent"], cmap="summer")
"""
Whatever sequence of numbers we pass into the c keyword argument will be converted to a range from 0 to 1.
These values will then be mapped onto a color map. Matplotlib has quite a few default colormaps.
In our case, we'll use the summer colormap, which results in green points when the associated number is low, and yellow when it's high."""
plt.show()




# Unfortunately, due to the number of schools, it's hard to interpret the map we made in the last screen.
"""
One way to make it easier to read very granular statistics is to aggregate them.
In this case, we can aggregate based on district, which will enable us to plot ell_percent district by district instead of school by school."""

grouped = combined.groupby("school_dist")
districts = grouped.aggregate(numpy.mean)
# Above two lines can be written as: districts = combined.groupby("school_dist").agg(numpy.mean)
# Reset the index of districts, making school_dist a column again.
districts.reset_index(inplace=True)

print(districts.head())




# Now that we've taken the mean of all the columns, we can plot out ell_percent by district.

m = Basemap(projection = "merc", llcrnrlat=40.496044,
    urcrnrlat=40.915256,
    llcrnrlon=-74.255735,
    urcrnrlon=-73.700272,
    resolution='i')

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.fillcontinents(color='white',lake_color='#85A6D9')

longitudes = districts["lon"].tolist()
latitudes = districts["lat"].tolist()

# s=50 to increase the size of the points in the scatterplot.
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=districts["ell_percent"], cmap="summer")

plt.show()






# There are several fields in combined that originally came from a survey of parents, teachers, and students.
# Make a bar plot of the correlations between these fields and sat_score.
correlations_sat_survey = combined.corr()["sat_score"][survey_columns].plot.bar()
plt.show()

""" Findings:
There are high correlations between N_s, N_t, N_p and sat_score.
Since these columns are correlated with total_enrollment, it makes sense that they would be high.
It is more interesting that rr_s, the student response rate, or the percentage of students that completed the survey,
correlates with sat_score. This might make sense because students who are more likely to fill out surveys
may be more likely to also be doing well academically.
How students and teachers percieved safety (saf_t_11 and saf_s_11) correlate with sat_score.
This make sense, as it's hard to teach or learn in an unsafe environment.
The last interesting correlation is the aca_s_11, which indicates how the student perceives academic standards,
correlates with sat_score, but this is not true for aca_t_11, how teachers perceive academic standards,
or aca_p_11, how parents perceive academic standards. """




# Safety And SAT Scores
combined.plot.scatter(x="sat_score", y="saf_s_11")
plt.show()
""" Findings:
There appears to be a correlation between SAT scores and safety, although it isn't that strong.
It looks like there are a few schools with extremely high SAT scores and high safety scores.
There are a few schools with low safety scores and low SAT scores.
No school with a safety score lower than 6.5 has an average SAT score higher than 1500 or so."""


# Map out safety scores.
m = Basemap(projection = "merc", llcrnrlat=40.496044,
    urcrnrlat=40.915256,
    llcrnrlon=-74.255735,
    urcrnrlon=-73.700272,
    resolution='i')

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.fillcontinents(color='white',lake_color='#85A6D9')

longitudes = districts["lon"].tolist()
latitudes = districts["lat"].tolist()

# s=50 to increase the size of the points in the scatterplot.
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=districts["saf_s_11"], cmap="summer")

plt.show()

""" Findings:
It looks like Upper Manhattan and parts of Queens and the Bronx tend to have lower safety scores, whereas Brooklyn has high safety scores."""


# Race And SAT Scores
race_columns = ["white_per", "asian_per", "black_per", "hispanic_per"]
combined.corr()["sat_score"][race_columns].plot.bar()
plt.show()
""" Findings:
It looks like a higher percentage of white or asian students at a school correlates positively with sat score,
whereas a higher percentage of black or hispanic students correlates negatively with sat score.
This may be due to a lack of funding for schools in certain areas,
which are more likely to have a higher percentage of black or hispanic students."""


combined.plot.scatter(x="hispanic_per", y="sat_score")
plt.show()


greater_than_95_hisp = combined[combined["hispanic_per"]>95]
print(greater_than_95_hisp["School Name"])

""" Findings:
The schools listed above appear to primarily be geared towards recent immigrants to the US.
These schools have a lot of students who are learning English, which would explain the lower SAT scores"""


less_than_10_hisp = combined[(combined["hispanic_per"]<10) & (combined["sat_score"]>1800)]
print(less_than_10_hisp["School Name"])

"""Findings:
Many of the schools above appear to be specialized science and technology schools that receive extra funding,
and only admit students who pass an entrance exam. This doesn't explain the low hispanic_per,
but it does explain why their students tend to do better on the SAT --
they are students from all over New York City who did well on a standardized test."""



# Gender And SAT Scores
gender_columns = ["male_per", "female_per"]
combined.corr()["sat_score"][gender_columns].plot.bar()
plt.show()

"""Findings:
In the plot above, we can see that a high percentage of females at a school positively correlates with SAT score,
whereas a high percentage of males at a school negatively correlates with SAT score. Neither correlation is extremely strong."""


combined.plot.scatter(x="female_per", y="sat_score")
plt.show()

""" Findings:
Based on the scatterplot, there doesn't seem to be any real correlation between sat_score and female_per.
However, there is a cluster of schools with a high percentage of females (60 to 80), and high SAT scores."""



print(combined["School Name"][(combined["female_per"]>60) & (combined["sat_score"]>1700)])

"""Findings:
These schools appears to be very selective liberal arts schools that have high academic standards."""




# AP Scores Vs SAT Scores

# Compute the percentage of students in each school that took the AP exam
combined["ap_per"] = combined["AP Test Takers "]/combined["total_enrollment"]
combined.plot.scatter(x="ap_per", y="sat_score")
plt.show()

"""Findings:
It looks like there is a relationship between the percentage of students in a school who take the AP exam,
and their average SAT scores. It's not an extremely strong correlation, though."""