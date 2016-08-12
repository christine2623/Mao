weight = [600,150,200,300,200,100,125,180]
height = [60,65,73,70,65,58,66,67]

# import the pyplot module in matplotlib
import matplotlib.pyplot as plt
# create a drawing figure and draw a plot using the scatter() method.
# height is plotted on the x-axis, and weight is plotted on the y-axis
plt.scatter(height, weight)
# show the plot with the show() method
plt.show()





import matplotlib.pyplot as plt
# Make a scatter plot with the wind column on the x-axis and the area column on the y-axis
plt.scatter(forest_fires["wind"], forest_fires["area"])
plt.show()
# Make a scatter plot with the temp column on the x-axis and the area column on the y-axis.
plt.scatter(forest_fires["temp"], forest_fires["area"])
plt.show()





"""Line charts are used when the observations are related in some way.
Let's say we wanted to graph the height of a single child as they aged."""
"""A line graph is laid out like a scatter plot, except each of the points is linked with a line."""

age = [5, 10, 15, 20, 25, 30]
height = [25, 45, 65, 75, 75, 75]
# Use the plot() method to plot age on the x-axis and height on the y-axis.
plt.plot(age, height)
plt.show()





# Bar graphs are used for communicating categorical information.
# see which areas in the park are the most flammable
area_by_y = forest_fires.pivot_table(index="Y", values="area", aggfunc=numpy.mean)
area_by_x = forest_fires.pivot_table(index="X", values="area", aggfunc=numpy.mean)
# Use the bar() method to plot area_by_y.index on the x-axis and area_by_y on the y-axis
plt.bar(area_by_y.index, area_by_y)
plt.show()

plt.bar(area_by_x.index, area_by_x)
plt.show()





"""
A horizontal bar graph is much like a regular bar graph, but the bars are horizontal instead of vertical.
This can be useful when communicating data that contains larger differences,
as there tends to be more horizontal space on pages than vertical space."""

# see how large the fires get at different times of the year
area_by_month = forest_fires.pivot_table(index="month", values="area", aggfunc=numpy.mean)
area_by_day = forest_fires.pivot_table(index="day", values="area", aggfunc=numpy.mean)
# Use the barh() method to plot range(len(area_by_month)) on the y-axis and area_by_month on the x-axis.
# The range() function will generate a sequence from 1 to the integer that is passed into it.
plt.barh(range(len(area_by_month)), area_by_month)
plt.show()

plt.barh(range(len(area_by_day)), area_by_day)
plt.show()







"""
Title -- the title() method.
X axis label -- the xlabel() method.
Y axis label -- the ylabel() method.
"""
# Make a scatter plot with the wind column of forest_fires on the x-axis and the area column of forest_fires on the y-axis.
plt.scatter(forest_fires["wind"], forest_fires["area"])
# Give the chart the title Wind speed vs fire area
plt.title("Wind speed vs fire")
# the y-axis label Area consumed by fire
plt.ylabel("Area consumed by fire")
# the x-axis label Wind speed when fire started.
plt.xlabel("Wind speed when fire started")
plt.show()






"""
fivethirtyeight -- the style of the plots on the site fivethirtyeight.com.
ggplot -- the style of the popular R plotting library ggplot.
dark_background -- will give the plot a darker background.
bmh -- the style used in a popular online statistics book.
"""
# Switch to the "fivethirtyeight" style
plt.style.use("fivethirtyeight")
plt.scatter(forest_fires["rain"], forest_fires["area"])
plt.show()