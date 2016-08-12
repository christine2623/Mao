import pandas as pd


all_ages = pd.read_csv("all_ages.csv")
all_ages.head(10)
recent_grads = pd.read_csv("recent_grads.csv")


# All values for Major_category
# print out "Major_category" column in DataFrame all_ages and do the value_counts() to count the same one altogether
# .index() means show only the index
print(all_ages['Major_category'].value_counts().index)

all_ages_major_categories = dict()
recent_grads_major_categories = dict()

# define a function to calculate the major categories
def calculate_major_cat_totals(df):
    # get a list of all major categories
    cats = df['Major_category'].value_counts().index
    # create an empty list
    counts_dictionary = dict()

    # iterate over the list of the category
    for c in cats:
        # if the category is the same, get the whole row
        major_df = df[df["Major_category"] == c]
        # sum up the "Total", “axis =0”means iterate through every row
        total = major_df["Total"].sum(axis=0)
        # store in the empty dict
        counts_dictionary[c] = total
    return counts_dictionary

# Call the function
all_ages_major_categories = calculate_major_cat_totals(all_ages)
# recent_grads_major_categories = calculate_major_cat_totals(recent_grads)




# Calculating how many college grads are unable to get higher wage

# Use the Low_wage_jobs and Total columns to calculate the proportion of recent college graduates that worked low wage jobs
# Store the resulting float as low_wage_percent.
low_wage_percent = 0.0
low_wage_percent = (recent_grads["Low_wage_jobs"].sum(axis=0))/(recent_grads["Total"].sum(axis=0))








# figure out all_ages or recent_grads has lower unemployment rate in certain major
# All majors, common to both DataFrames
majors = recent_grads['Major'].value_counts().index

recent_grads_lower_unemp_count = 0
all_ages_lower_unemp_count = 0

for maj in majors:
    all_ages_row = all_ages[all_ages["Major"] == maj]
    recent_grads_row = recent_grads[recent_grads["Major"] == maj]

    # recent_grads[recent_grads["Major"]==major]["Unemployment_rate"] returns a Series.
    # The .values attribute of a series returns another series Documentation whereas .values[0] extracts the value.
    recent_grads_unemp_rate = recent_grads_row['Unemployment_rate'].values[0]
    all_ages_unemp_rate = all_ages_row['Unemployment_rate'].values[0]

    if all_ages_unemp_rate > recent_grads_unemp_rate:
        recent_grads_lower_unemp_count += 1
    if all_ages_unemp_rate < recent_grads_unemp_rate:
        all_ages_lower_unemp_count += 1





