# coding: utf-8

# In[1]:

import sys
import os

print(sys.executable)
print(os.getcwd())
os.chdir("C:\\Users\\Christine\\PycharmProjects\\test\\")
print(os.getcwd())

# In[3]:

import pandas as pd

police_killings = pd.read_csv("police_killings.csv", encoding="ISO-8859-1")
print(police_killings.head())

# In[5]:

print(police_killings.columns)

# In[7]:

print(police_killings["raceethnicity"].value_counts())

# In[20]:

import matplotlib.pyplot as plt
import seaborn as sns

# get_ipython().magic('matplotlib inline')

counts = police_killings["raceethnicity"].value_counts()
print(counts.index)

# plt.bar(location of bars, what to be shown)
plt.bar(range(6), counts)
# plt.xticks(location of ticks, labels of the ticks, rotation?, color)
plt.xticks(range(6), counts.index, rotation="vertical", color="white")
plt.yticks(color="white")
plt.show()

# Most of them are Whites. Then black. Does it has something to do with the percentage of each race in the US population?
# 

# In[23]:

ethicity_ratio = counts / sum(counts)
print(ethicity_ratio)

# Racial breakdown
# It looks like people identified as Black are far overrepresented in the shootings versus in the population of the US (28% vs 16%). You can see the breakdown of population by race on wikipedia.
# People identified as Hispanic appear to be killed about as often as random chance would account for (14% of the people killed as Hispanic, versus 17% of the overall population).
# Whites are underrepresented among shooting victims vs their population percentage, as are Asians.

# In[34]:

# with higher bins, you will get more bars. (granular result)
police_killings["p_income"][police_killings["p_income"] != "-"].astype(int).hist(bins=25)
plt.xticks(color="white")
plt.yticks(color="white")
plt.xlabel("dollars", color="white")
plt.ylabel("counts", color="white")
plt.title("median personal income by census area", color="white")

# Income breakdown
# According to the Census, median personal income in the US is 28,567, and our median is 22,348, which means that shootings tend to happen in less affluent areas. Our sample size is relatively small, though, so it's hard to make sweeping conclusions.

# In[39]:

state_pop = pd.read_csv("population.csv")

# In[40]:

count = police_killings["state_fp"].value_counts()

# In[45]:

# Create a new Dataframe called states. One column should be called STATE, and will contain the index of counts. 
# The other column should be called shootings, and will contains the values from counts
states = pd.DataFrame({"STATE": count.index, "shooting": count})

# In[47]:

# Use the merge() method to merge state_pop and states
# Pass the on keyword argument, with the value set to STATE. 
# STATE is the common column that both states and state_pop share.
states = states.merge(state_pop, on="STATE")

# In[67]:

states["pop_millions"] = states["POPESTIMATE2015"] / 1000000
states["rate"] = states["shooting"] / states["pop_millions"]

states.sort_values(by="rate")
print(states)

#
# Killings by state
# States in the midwest and south seem to have the highest police killing rates, whereas those in the northeast seem to have the lowest

# In[56]:

pk = police_killings[(police_killings["share_white"] != "-") & (police_killings["share_black"] != "-") &
                     (police_killings["share_hispanic"] != "-")]

pk["share_white"] = pk["share_white"].astype(float)
pk["share_black"] = pk["share_black"].astype(float)
pk["share_hispanic"] = pk["share_hispanic"].astype(float)

# In[58]:

police_killings["state"].value_counts()

# In[60]:

lowest_states = ["CT", "PA", "IA", "NY", "MA", "NH", "ME", "IL", "OH", "WI"]
highest_states = ["OK", "AZ", "NE", "HI", "AK", "ID", "NM", "LA", "CO", "DE"]

# use isin() to only select rows from police_killings where the state column is in the list.
ls = pk[pk["state"].isin(lowest_states)]
hs = pk[pk["state"].isin(highest_states)]

# In[65]:

columns = ["pop", "county_income", "share_white", "share_black", "share_hispanic"]

print(ls[columns].mean())
print(hs[columns].mean())


# 
# State by state rates
# 
# It looks like the states with low rates of shootings tend to have a higher proportion of blacks in the population, and a lower proportion of hispanics in the census regions where the shootings occur. It looks like the income of the counties where the shootings occur is higher.
# States with high rates of shootings tend to have high hispanic population shares in the counties where shootings occur.

# In[ ]:
