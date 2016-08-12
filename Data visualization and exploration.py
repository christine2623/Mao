"""Data visualization is the dominant technique within data exploration that allows you to develop some initial hypotheses for the relationships
between variables and some general trends that will help you navigate your data workflow better. """

import pandas as pd
recent_grads = pd.read_csv('recent-grads.csv')




#  .hist() method specifies a column parameter to specify the column(s) we want a histogram for.
import matplotlib.pyplot as plt

columns = ['Median','Sample_size']
recent_grads.hist(column=columns)


# Set the `layout` parameter as `(2,1)` so the graphs are displayed as 2 rows & 1 column
# Then set `grid` parameter to `False`.
# using 50 bins instead of the default 10
recent_grads.hist(column=columns, layout=(2,1), grid=False, bins = 50)






# Select just `Sample_size` & `Major_category` columns from `recent_grads`
# Name the resulting DataFrame as `sample_size`
sample_size = recent_grads[['Sample_size', 'Major_category']]

# Run the `boxplot()` function on `sample_size` DataFrame and specify, as a parameter,
# that we'd like a box and whisker diagram to be generated for each unique `Major_category`
sample_size.boxplot(by='Major_category')

# Format the resulting plot to make the x-axis labels (each `Major_category` value)
# appear vertically instead of horizontally (by rotating 90 degrees)
plt.xticks(rotation=90)





# Use color to differentiate multiple plots in one chart
# Plot Unemployment_rate on x-axis, Median salary on y-axis, in red
plt.scatter(recent_grads['Unemployment_rate'], recent_grads['Median'], color='red')
# Plot ShareWomen (Female % in major) on x-axis, Median salary on y-axis, in blue
plt.scatter(recent_grads['ShareWomen'], recent_grads['Median'], color='blue')
plt.show()






