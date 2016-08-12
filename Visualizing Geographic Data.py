"""
map projection can  convert the values from the spherical coordinate system (which is in 3 dimensions) to the cartesian coordinate system (which is in 2 dimensions)
"""

"""
The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python.
Basemap does not do any plotting on it’s own,
but provides the facilities to transform coordinates to one of 25 different map projections.
"""

# conda install basemap
from mpl_toolkits.basemap import Basemap

"""
To create a new instance of the Basemap class, The following parameters are required:
projection - the map projection.
llcrnrlat - latitude of lower left hand corner of the desired map domain (degrees).
urcrnrlat - latitude of upper right hand corner of the desired map domain (degrees).
llcrnrlon - longitude of lower left hand corner of the desired map domain (degrees).
urcrnrlon- longitude of upper right hand corner of the desired map domain (degrees).
"""

# Create a new Basemap instance
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# %matplotlib inline

m = Basemap(projection = "merc", llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180)

# Convert from Series objects to List objects.
longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()

# Convert latitude and longitude to x and y coordinates.
x, y = m(longitudes, latitudes)

# Display original longitude values
print(longitudes[0:5])
# Display original latitude values
print(latitudes[0:5])
# Display x-axis coordinates
print(x[0:5])
# Display y-axis coordinates
print(y[0:5])

# create a scatter plot. Use the s parameter to specify the size of each dot to be 1
m.scatter(x, y, s=1)



# Use the drawcoastlines method to draw the coast lines
m.drawcoastlines()
plt.show()




# Create a Figure with a figsize of 15 inches by 20 inches
fig = plt.figure(figsize=(15,20))
# Create an Axes object by using the add_subplot() method
ax1 = fig.add_subplot(1,1,1)
# Call the set_title() method on the Axes object to set the title to "Scaled Up Earth With Coastlines".
ax1.set_title("Scale Up Earth With Coastlines")

m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()
x, y = m(longitudes, latitudes)

m.scatter(x, y, s=1)
m.drawcoastlines()
plt.show()





"""
 drawgreatcircle() to display a great circle between 2 points
 requires 4 parameters
 lon1 - longitude of the starting point.
lat1 - latitude of the starting point.
lon2 - longitude of the ending point.
lat2 - latitude of the ending point.
"""

fig = plt.figure(figsize=(15,20))
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
m.drawcoastlines()

# written a function called create_great_circles() that takes in a DataFrame and calls the drawgreatcircle() method to plot each route
def create_great_circles(df):
    # df.iterrows() iterates over DataFrame rows as (index, Series) pairs.
    for index, row in df.iterrows():
        start_lon = row['start_lon']
        start_lat = row['start_lat']
        end_lon = row['end_lon']
        end_lat = row['end_lat']
        # draw the circle if it is the shorter side!
        if abs(end_lat - start_lat) < 180 and abs(end_lon - start_lon) < 180:
            # Use drawgreatcircle method to draw the circle
            m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, linewidth=1)

# Create a filtered DataFrame containing just the routes that start at the DFW airport.
dfw = geo_routes[geo_routes["source"] == "DFW"]
# Call the function to draw the circle
create_great_circles(dfw)

plt.show()


