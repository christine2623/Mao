import pandas as pd
# We need to specify an encoding because the dataset has some characters that aren't in the Python default utf-8 encoding.
star_wars = pd.read_csv("StarWars.csv", encoding="ISO-8859-1")

# Drop all the rows if it has NaN in "RespondentID" column
print(star_wars.shape[0])
print(star_wars.columns)
# star_wars = star_wars.dropna(subset=["RespondentID"])
pd.notnull(star_wars["RespondentID"])
print(star_wars.shape[0])


# If you have series object, you can use dictionary to do the mapping from each value in series to a new value
# Switch the "Yes" to True, "No" to Fasle, Booleans are easier to work with because you can select the rows that are
# True or False without having to do a string comparisons.
all_films_list = star_wars["Have you seen any of the 6 films in the Star Wars franchise?"]
# You can use a dictionary to define a mapping from each value in series to a new value
yes_no ={"Yes": True, "No": False}
fan_of_star_wars = star_wars["Do you consider yourself to be a fan of the Star Wars film franchise?"]
# use the map() method on Series to do this conversion.
all_films_list = all_films_list.map(yes_no)
fan_of_star_wars = fan_of_star_wars.map(yes_no)


# checkbox if you seen the movie or not. Modify the checkbox to True and False fo every star wars movie.
# Create a dictionary to define a mapping to each values in series to a new value.
Episodes = {"Star Wars: Episode I  The Phantom Menace": True, "Star Wars: Episode II  Attack of the Clones": True, "Star Wars: Episode III  Revenge of the Sith": True, "Star Wars: Episode IV  A New Hope": True, "Star Wars: Episode V The Empire Strikes Back": True, "Star Wars: Episode VI Return of the Jedi": True, "NaN": False}
# perform the conversion
star_wars["Which of the following Star Wars films have you seen? Please select all that apply"]=star_wars["Which of the following Star Wars films have you seen? Please select all that apply."].map(Episodes)
star_wars["Unnamed: 4"]=star_wars["Unnamed: 4"].map(Episodes)
star_wars["Unnamed: 5"]=star_wars["Unnamed: 5"].map(Episodes)
star_wars["Unnamed: 6"]=star_wars["Unnamed: 6"].map(Episodes)
star_wars["Unnamed: 7"]=star_wars["Unnamed: 7"].map(Episodes)
star_wars["Unnamed: 8"]=star_wars["Unnamed: 8"].map(Episodes)
# Faster: star_wars[star_wars.columns[3:9]] = star_wars[star_wars.columns[3:9]].map(Episodes)

# Rename the column to a readable name
# use dict to do this, and rename() function
star_wars = star_wars.rename(columns={"Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1"})
star_wars = star_wars.rename(columns={"Unnamed: 4": "seen_2"})
star_wars = star_wars.rename(columns={"Unnamed: 5": "seen_3"})
star_wars = star_wars.rename(columns={"Unnamed: 6": "seen_4"})
star_wars = star_wars.rename(columns={"Unnamed: 7": "seen_5"})
star_wars = star_wars.rename(columns={"Unnamed: 8": "seen_6"})


# Change the type of the value for column 9 to 14 to float
star_wars[star_wars.columns[9:15]][1:] = star_wars[star_wars.columns[9:15]][1:].astype(float)
# Rename the column
star_wars = star_wars.rename(columns={"Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1"})
star_wars = star_wars.rename(columns={"Unnamed: 10": "ranking_2"})
star_wars = star_wars.rename(columns={"Unnamed: 11": "ranking_3"})
star_wars = star_wars.rename(columns={"Unnamed: 12": "ranking_4"})
star_wars = star_wars.rename(columns={"Unnamed: 13": "ranking_5"})
star_wars = star_wars.rename(columns={"Unnamed: 14": "ranking_6"})
# Using for loop to rename:
"""
column_titles = star_wars.columns[9:15]
count = 1
for col in column_titles:
    star_wars = star_wars.rename(columns={col:"ranking_"+str(count)})
    count+=1
    """


# Use the mean method to compute the mean of each of the ranking columns from the last screen.
# Make a bar chart of each ranking.
mean_ranking = star_wars[star_wars.columns[9:15]][1:].mean()

import matplotlib.pyplot as plt
# %matplotlib inline

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())
plt.show()

"""Findings:
As predicted in the beggining of the instructions, Episode V is the the most popular.
Additionally, all of the original trilogy (IV, V, VI) scored much better than the newer three."""


# Use the sum() to see which movie is the most seen one.
plt.bar(range(6), star_wars[star_wars.columns[3:9]].sum())
plt.show()

"""Findings:
Not surprisingly there seems to be a strong correlation between the ranking of the movie and how many times the movie had been seen."""


# Exploring The Data By Binary Segments

A_fan = star_wars[star_wars["Do you consider yourself to be a fan of the Star Wars film franchise?"]==True]
Not_a_fan = star_wars[star_wars["Do you consider yourself to be a fan of the Star Wars film franchise?"]==False]

plt.bar(range(6), A_fan[A_fan.columns[9:15]].mean())
plt.show()

plt.bar(range(6), Not_a_fan[Not_a_fan.columns[9:15]].mean())
plt.show()

plt.bar(range(6), A_fan[A_fan.columns[3:9]].sum())
plt.show()

plt.bar(range(6), Not_a_fan[Not_a_fan.columns[3:9]].sum())
plt.show()

"""Findings:
More males than females saw the newer three movies but on average liked the movies less.
Phantom Menace was much better received by female viewers, but still was liked less than the original trilogy. """


# Another way to do the plotting
import numpy as np

def avg_rankings(df):
    avg = []
    for col in df.columns[9:15]:
        avg.append(df[col].mean())
    return avg


def seen_list(df):
    seen = []
    for i in range(1,7):
        seen.append(df["seen_"+str(i)].sum())
    return seen

males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]

index = np.arange(len(avg_rankings(males)))
bar_width = 0.35

plt.bar(index,avg_rankings(males), bar_width, color='b')
plt.bar(index+bar_width, avg_rankings(females), bar_width, color='g')
plt.title("Mean Star Wars Film Rankings by Gender")
plt.show()


index = np.arange(len(seen_list(males)))

plt.bar(index, seen_list(males), bar_width, color='b')
plt.bar(index+bar_width, seen_list(females), bar_width, color='g')
plt.title("Count of Movies Seen by Gender")
plt.show()






# Another way to do the plotting

import numpy as np

n_groups = 6
index =  np.arange(n_groups)
opacity = 0.4
bar_width = 0.35

fig, ax = plt.subplots()
ax.bar(index,mean_ranking,bar_width,alpha=opacity,color='y',label="Average")

plt.xlabel('Star Wars Films')
plt.ylabel('Ranking')
plt.title('Ranking for the Star Wars Films')
plt.xticks(index + bar_width,['Episode_'+str(i) for i in range(1,7)])
plt.tight_layout()
plt.legend()



# Use the sum method to compute the sum of each of the seen columns from a previous screen.
# Make a bar chart of each ranking. You can use a matplotlib bar chart for this.

sum_seen = star_wars[star_wars.columns[3:9]].sum() # number of people who had seen a movie i

fig, ax = plt.subplots()
ax.bar(index,sum_seen,bar_width,alpha=opacity,color='b',label="Sum")

plt.xlabel('Star Wars Films')
plt.ylabel('Number of People')
plt.title('Audience')
plt.xticks(index + bar_width,['Episode_'+str(i) for i in range(1,7)])
plt.tight_layout()
plt.legend()







#Splitting Data into two groups : Male/Female

males = star_wars[star_wars["Gender"] == "Male"]

# Most seen Star Wars film among Males

sum_males = males.iloc[:,3:9].sum()

# Most seen Star Wars film among Males

females = star_wars[star_wars["Gender"] == "Female"]

unisex_audience = len(males) + len(females)

print("People who watched the Star Wars films: "+ str(unisex_audience))

sum_females = females.iloc[:,3:9].sum()

fig, ax = plt.subplots()
ax.bar(index,sum_males,bar_width,alpha=opacity,color='b',label="Males")

plt.xlabel('Star Wars Films')
plt.ylabel('Number of People')
plt.title('Audience Female/Male')
plt.xticks(index + bar_width,['Episode_'+str(i) for i in range(1,7)])
plt.tight_layout()
plt.legend()

ax.bar(index + bar_width,sum_females,bar_width,alpha=opacity,color='y',label="Females")


plt.legend()