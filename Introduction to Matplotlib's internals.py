"""
So far I've learned plotting methods.
Matplotlib's high-level plotting methods - e.g. .scatter(), .plot()
Seaborn's high-level plotting methods - e.g. .distplot(), .boxplot()
Pandas DataFrame methods - e.g. .hist(), .boxplot()
"""
import matplotlib.pyplot as plt

# 2 simple lists of values
month = [1,1,2,2,4,5,5,7,8,10,10,11,12,12]
temperature = [32,15,40,35,50,55,52,80,85,60,57,45,35,105]
# month on the x-axis, temperature on the y-axis
plt.scatter(month, temperature)
plt.show()





"""
Figure is the top-level Matplotlib object that manages the entire plotting area.
A Figure instance acts as a container for your plots and contains some useful parameters and methods like:
The figsize(w,h) parameter lets you specify the width w and height h, in inches, of the plotting area
The dpi parameter lets you specify the density, in dots per inch
The .add_subplot() method lets you add individual plots to the Figure instance
"""
# Subplot is the Matplotlib object that you use to create the axes for a plot

import numpy as np
month = [1,1,2,2,4,5,5,7,8,10,10,11,12,12]
temperature = [32,15,40,35,50,55,52,80,85,60,57,45,35,105]

# call plt.figure() to instantiate a new Figure instance (width: 5 inches, height: 7 inches)
fig = plt.figure(figsize=(5,7))
# call .add_subplot(1,1,1) on the Figure instance to add an empty plot
# the first parameter refers to the row number 1; the second parameter refers to the column number 2; the third parameter refers to the nth plot in the Figure to be returned (only 1 plot in this case)
# fig.add_subplot(nrows, ncols, plot_number)
ax = fig.add_subplot(1,1,1)

# Set the x-axis ticks to range from the lowest value in month to the highest value in month
ax.set_xlim([np.min(month), np.max(month)])
ax.set_ylim([np.min(temperature), np.max(temperature)])
# change the x-axis ticks to range from 0 to 13
ax.set_xlim([0,13])

# set the x-axis label as "Month" using the method: ax.set_xlabel()
ax.set_xlabel("Month")
ax.set_ylabel("Temperature")
ax.set_title("Year Round Temperature")

# An easier way to set the limit, labels, title and everything
ax.set(xlim=(0,13), ylim=(10,110) , xlabel = "Month", ylabel = "Temperature", title = "Year Round Temperature")
ax.scatter(month, temperature, color='darkblue', marker='o')

# run the .scatter() method, params: color, marker
ax.scatter(month, temperature, color="darkblue", marker="o")
plt.show()

# Print the types
print(type(fig))
print(type(ax))







"""
When adding multiple subplots to the same Figure instance by calling .add_subplot() each time for each plot you want,
the first 2 parameters remain the same and only the third parameter, plot_number,
changes based on which plot you want returned from that specific function call.
To place 2 subplots on the plotting area vertically, where one plot is above the other,
you'll need to specify that you want a grid with 2 rows and 1 column,
and then to return plot_number 1 the first time and plot_number 2 the second time:
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
"""
"""
if you have a grid with 2 rows and 2 columns, the 2nd plot (plot_number = 2) is located at row 1 and column 2
while the 3rd plot (plot_number = 3) is located at row 2 and column 1"""

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)






month_2013 = [1,2,3,4,5,6,7,8,9,10,11,12]
temperature_2013 = [32,18,40,40,50,45,52,70,85,60,57,45]
month_2014 = [1,2,3,4,5,6,7,8,9,10,11,12]
temperature_2014 = [35,28,35,30,40,55,50,71,75,70,67,49]

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter(month_2013, temperature_2013, color = "darkblue", marker = "o")
ax2.scatter(month_2014, temperature_2014, color = "darkgreen", marker= "o")
ax1.set(xlim = (0,13), ylim = (10,110), title = "2013")
ax2.set(xlim = (0,13), ylim = (10,110), title = "2014")
plt.show()