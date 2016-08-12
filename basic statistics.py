# equal interval
# logarithmic scale

car_speeds = [10,20,30,50,20]
earthquake_intensities = [2,7,4,5,8]

# Get tge mean of the above lists
mean_car_speed = sum(car_speeds)/len(car_speeds)
# This value will not be meaningful, because we shouldn't average values on a logarithmic scale this way.
mean_earthquake_intensities = sum(earthquake_intensities)/len(earthquake_intensities)




# Scales can be discrete or continuous
# discrete: numbers of cars
# continuous: inches

day_numbers = [1,2,3,4,5,6,7]
snail_crawl_length = [.5,2,5,10,1,.25,4]
cars_in_parking_lot = [5,6,4,2,1,7,8]

import matplotlib.pyplot as plt

# Make a line plot with day_numbers on the x axis and snail_crawl_length on the y axis.
plt.plot(day_numbers, snail_crawl_length)
plt.show()
# Make a line plot with day_numbers on the x axis and cars_in_parking_lot on the y axis.
plt.plot(day_numbers, cars_in_parking_lot)
plt.show()




