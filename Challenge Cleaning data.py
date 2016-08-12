import pandas as pd

avengers = pd.read_csv("avengers.csv")
avengers.head(5)






import matplotlib.pyplot as plt
%matplotlib inline
true_avengers = pd.DataFrame()

# Avengers has been created since 1960 so we want to drop the avengers which has created before 1960
avengers['Year'].hist()
true_avengers = avengers[avengers["Year"]>=1960]
true_avengers["Year"].hist()






# Create a new column, Deaths, that contains the number of times each superhero died.
# define a function to get the value of the new column "Deaths"
def deaths_count(df):
    death_count = 0
    columns = ['Death1', 'Death2', 'Death3', 'Death4', 'Death5']
    for c in columns:
        death = df[c]
        if pd.isnull(death) or death == "NO":
            continue
        elif death == "YES":
            death_count += 1
    return death_count

true_avengers["Deaths"] = true_avengers.apply(deaths_count, axis=1)





joined_accuracy_count = 0
true_avengers["accurate_joined_year"] = 2015 - true_avengers["Year"]
# use index to perform the iteration in dataframe
for row_index in range(0, true_avengers.shape[0]):
    if true_avengers.iloc[row_index]["accurate_joined_year"] == true_avengers.iloc[row_index]["Years since joining"]:
        joined_accuracy_count += 1