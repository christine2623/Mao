import pandas as pd
avengers = pd.read_csv("avengers.csv")
true_avengers = pd.DataFrame()
joined_accuracy_count = int()

# print(avengers.head())
true_avengers = avengers[avengers["Year"]>=1960]
# print(true_avengers.head())

# Method one to get the counts of the correct "Years since joining"
joined_accuracy_count  = int()
correct_joined_years = true_avengers[true_avengers['Years since joining'] == (2015 - true_avengers['Year'])]
joined_accuracy_count = len(correct_joined_years)

# Method two to get the counts of the correct "Years since joining"
joined_accuracy_count = 0
true_avengers["accurate_joined_year"] = 2015 - true_avengers["Year"]
# print(true_avengers.head())
for row_index in range(0, true_avengers.shape[0]):
    print(row_index)
    # print(true_avengers.iloc[row_index]["accurate_joined_year"])
    if true_avengers.iloc[row_index]["accurate_joined_year"] == true_avengers.iloc[row_index]["Years since joining"]:
        joined_accuracy_count += 1

print("end")
for row_index in range(0, 5):
    print(row_index)
row_index = 0
while row_index < 5: # 5 is length or size of list
    print(row_index) # list[index], javaarraylist.get(index), panda.iloc[index]
    row_index += 1
print("end")
print(avengers.shape[0])
print(true_avengers.shape[0])
print(joined_accuracy_count)
