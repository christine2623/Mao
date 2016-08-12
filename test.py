import pandas as pd

fandango = pd.read_csv('fandango_score_comparison.csv')
first_row = 0
last_row = fandango.shape[0] - 1
# Return a DataFrame containing just the first and the last row and assign to first_last.
output = pd.DataFrame([])
output = output.append(fandango.iloc[first_row])
output = output.append(fandango.iloc[last_row], ignore_index=True)
print(output.transpose().to_string())
print(output.to_string())
print("zzz")

print(fandango.iloc[[first_row,last_row],[0, 5]].to_string())