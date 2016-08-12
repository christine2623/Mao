# Import the Pandas library.
import pandas
# Read food_info.csv into a DataFrame object named food_info
food_info = pandas.read_csv("food_info.csv")
# Use the columns attribute followed by the tolist() method to return a list containing just the column names.
cols = food_info.columns.tolist()
print(cols)
# Display the first 3 rows of food_info.
print(food_info.head(3))


"""# Adds 100 to each value in the column and returns a Series object.
add_100 = food_info["Iron_(mg)"] + 100

# Subtracts 100 from each value in the column and returns a Series object.
sub_100 = food_info["Iron_(mg)"] - 100

# Multiplies each value in the column by 2 and returns a Series object.
mult_2 = food_info["Iron_(mg)"]*2"""




# Assign the number of grams of protein per gram of water ("Protein_(g)" column divided by "Water_(g)" column)
# to grams_of_protein_per_gram_of_water.
grams_of_protein_per_gram_of_water = food_info["Protein_(g)"] / food_info["Water_(g)"]
# Assign the total amount of calcium and iron ("Calcium_(mg)" column plus "Iron_(mg)" column) to milligrams_of_calcium_and_iron
milligrams_of_calcium_and_iron = food_info["Calcium_(mg)"] + food_info["Iron_(mg)"]




# Multiply the "Protein_(g)" column by 2 and assign the resulting Series to weighted_protein.
weighted_protein = food_info["Protein_(g)"] *2
# Multiply the "Lipid_Tot_(g)" column by -0.75 and assign the resulting Series to weighted_fat.
weighted_fat = food_info["Lipid_Tot_(g)"] * -0.75
# Add both Series objects together and assign to initial_rating.
initial_rating = weighted_protein + weighted_fat





"""While there are many ways to normalize data, one of the simplest ways is
to divide all of the values in a column by that column's maximum value.
This way, all of the columns will range from 0 to 1.
To calculate the maximum value of a column, use the Series method max().
In the following code, we use the max() method to calculate the largest value in the "Energ_Kcal" column and assign it to max_calories:

# The largest value in the "Energ_Kcal" column.
max_calories = food_info["Energ_Kcal"].max()

You can then use the division arithmetic operator (/)
to divide the values in the "Energ_Kcal" column by the maximum value, max_calories:

# Divide the values in "Energ_Kcal" by the largest value to normalize.
normalized_calories = food_info["Energ_Kcal"] / max_calories
"""

# Normalize the values in the "Lipid_Tot_(g)" column and assign the result to normalized_fat.
# Get the max of the column
max_lipid = food_info["Lipid_Tot_(g)"].max()
# all of the columns will range from 0 to 1
normalized_fat = food_info["Lipid_Tot_(g)"] / max_lipid
print(normalized_fat[0:5])






"""iron_grams = food_info["Iron_(mg)"] / 1000
# Assign the value iron_grams to the dataframe food_info
food_info["Iron_(g)"] = iron_grams
# The DataFrame food_info now has the "Iron_(g)" column, which contains the values from iron_grams"""

# Normalize the "Protein_(g)" column and add it to food_info as the "Normalized_Protein" column.
protein_max = food_info["Protein_(g)"].max()
Normalized_Protein = food_info["Protein_(g)"]/protein_max
food_info["Normalized_Protein"] = Normalized_Protein

# Normalize the "Lipid_Tot_(g)" column and add it to food_info as the "Normalized_Fat" column.
normalized_fat = food_info["Lipid_Tot_(g)"] / food_info["Lipid_Tot_(g)"].max()
food_info["Normalized_Fat"] = normalized_fat




# Perform Score=2×(Normalized_Protein)−0.75×(Normalized_Fat)
# Normalized the Protein column and the fat column so they can have the same effect on the result purely because the scale of the values
food_info["Normalized_Protein"] = food_info["Protein_(g)"] / food_info["Protein_(g)"].max()
food_info["Normalized_Fat"] = food_info["Lipid_Tot_(g)"] / food_info["Lipid_Tot_(g)"].max()
# Score=2×(Normalized_Protein)−0.75×(Normalized_Fat)
norm_nutr_index = (food_info["Normalized_Protein"] *2) + (food_info["Normalized_Fat"]* -0.75)
# create a new column for it in the DataFrame food_info
food_info["Norm_Nutr_Index"] = norm_nutr_index






# If we want to explore which foods rank the best on the Norm_Nutr_Index column, we need to sort the DataFrame by that column.
# DataFrame objects contain a sort_values() method that we can use to sort the entire DataFrame by.
"""To sort the DataFrame on the Sodium_(mg) column, pass in the column name to the sort_values() method
and assign the returned DataFrame to a new variable"""
# food_info.sort_values("Sodium_(mg)")
"""By default, Pandas will sort by the column we specified in ascending order
and will return a new DataFrame instead of modifying food_info itself."""
"""
# The DataFrame is sorted in-place instead of returning a new DataFrame.
food_info.sort_values("Sodium_(mg)", inplace=True)
# Sort in descending order this time instead of ascending.
food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)"""





food_info["Normalized_Protein"] = food_info["Protein_(g)"] / food_info["Protein_(g)"].max()
food_info["Normalized_Fat"] = food_info["Lipid_Tot_(g)"] / food_info["Lipid_Tot_(g)"].max()
food_info["Norm_Nutr_Index"] = 2*food_info["Normalized_Protein"] + (-0.75*food_info["Normalized_Fat"])
# Sort the food_info DataFrame in-place on the Norm_Nutr_Index column in descending order.
food_info.sort_values("Norm_Nutr_Index", inplace= True, ascending = False)
