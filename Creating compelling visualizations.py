"""
Seaborn is a Python library supported by Stanford University that enables you to create beautiful, presentation-ready data visualizations.
"""

import pandas as pd
births = pd.read_csv("births.csv")

# .distplot() to generate a histogram of values in the prglngth column, the length of each pregnancy in weeks.
# recommend using sns.plt.show() to distinguish from Matplotlib plots


import matplotlib.pyplot as plt
import seaborn as sns
#  kernel density estimate = False
sns.distplot(births['prglngth'], kde=False)
sns.plt.show()




#  Seaborn was imported to the environment and overrode the default styles.
import seaborn as sns
births['agepreg'].hist()
sns.plt.show()





sns.distplot(births['prglngth'], kde=False)
sns.axlabel('Pregnancy Length, weeks', 'Frequency')
sns.plt.show()



# Plot a histogram of the birthord column
# style: "dark"
sns.set_style('dark')
sns.distplot(births['birthord'], kde=False)
# x-axis label: Birth number. y-axis label: Frequency.
sns.axlabel('Birth number', 'Frequency')
sns.plt.show()




# generate a boxplot with the birthord column on the x-axis and the agepreg column on the y-axis
births = pd.read_csv('births.csv')
sns.boxplot(x='birthord', y='agepreg', data=births)
sns.plt.show()





"""
Whenever a variable is plotted against a different variable (e.g. var1 vs var2),
Seaborn generates a scatter plot with var1 on the x-axis and var2 on the y-axis.
Whenever a variable is plotted against itself (e.g. var1 vs var1), however,
Seaborn generates a histogram of values instead,
since a scatter plot where the axes are the same variable isn't particularly useful!
"""
sns.pairplot(data = births, vars = ["agepreg","prglngth", "birthord"])
sns.plt.show()