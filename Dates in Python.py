"""The time module deals primarily with Unix timestamps.
A Unix timestamp is a simple floating point value, with no explicit mention of day, month, or year.
This floating point value represents the number of seconds that have passed since the epoch.
The epoch is the first second of the year 1970. So, a timestamp of 0.0 would represent the epoch, and a timestamp of 60.0 would represent one minute after the epoch.
Any date after 1970 can be represented this way."""

# To get the Unix timestamp for the current time, we use the time() function within the time module

import time
# assign the timestamp for the current time to current_time
current_time = time.time()




# The gmtime() function takes a timestamp as an argument, and returns an instance of the struct_time class.
"""tm_year: The year of the timestamp
tm_mon: The month of the timestamp (1-12)
tm_mday: The day in the month of the timestamp (1-31)
tm_hour: The hour of the timestamp (0-23)
tm_min: The minute of the timestamp (0-59)"""

import time
# assign the current Unix timestamp to current_time
current_time = time.time()
# convert current_time to a struct_time object
current_struct_time = time.gmtime(current_time)
# Assign the tm_hour property of current_struct_time to current_hour
current_hour = current_struct_time.tm_hour
print(current_hour)





# The time module deals primarily with timestamps in UTC
"""The datetime module has better support for working extensively with dates.
With datetime, it is easier to work with different time zones and perform arithmetic (adding days, for example) on dates."""
# The datetime module contains a datetime class to represent points in time.
"""Following property:
year
month
day
hour
minute
second
microsecond"""

"""To get the datetime instance representation of the current time, the datetime class has a now() class method.
Class methods are called on the class itself, so we would write datetime.datetime.now() to create a datetime instance representing the current time.
The first datetime is the module, .datetime is the class, and .now() is the class method."""

# Import the datetime module
import datetime
# Assign the datetime object representation of the current time to current_datetime.
current_datetime = datetime.datetime.now()
# Assign the year property of current_datetime to current_year.
current_year = current_datetime.year
current_month = current_datetime.month



"""datetime module provides the timedelta class: perform arithmetic on date"""
"""When we instantiate instances of the timedelta class, we can specify the following parameters:
weeks
days
hours
minutes
seconds
milliseconds
microseconds"""

# we wanted to get the date that is 3 weeks and 2 days from today.
# first get an instance of the datetime class to represent today:
today = datetime.datetime.now()
# get an instance of the timedelta class to represent the span of time we are working with:
diff = datetime.timedelta(weeks = 3, days = 2)
# add:
result = today + diff



import datetime
# Create an instance of the datetime class to represent the current time and date.
today = datetime.datetime.now()
# Create an instance of the timedelta class to represent one day.
diff = datetime.timedelta(days = 1)

tomorrow = today + diff
yesterday = today - diff




"""datetime class's instance method called strftime(). strftime() takes a formatted string as its input.
Format strings contain special indicators, usually preceded by a "%" character,
that indicate where a certain value should go."""

"""suppose we store the date March 3, 2010 into the object march3.
If we want to format it nicely into the string "Mar 03, 2010", we can do that:
march3 = datetime.datetime(year = 2010, month = 3, day = 3)
pretty_march3 = march3.strftime("%b %d, %Y")
print(pretty_march3)
"""
# the abbreviated month name ("%b")
# the day in the month ("%d")
# the full year ("%Y")



# March 3, 2010 at 11:00AM would look like "11:00AM on Wednesday March 03, 2010" in this format
mystery_date_formatted_string = mystery_date.strftime("%I:%M%p on %A %B %d, %Y")
print(mystery_date_formatted_string)
"""you'd never be expected to memorize which indicators have what meaning,
so the strftime() documentation has a useful table that will help you format dates"""




"""convert a formatted string into a datetime object.
The datetime class (datetime.datetime) contains a class method called strptime() which takes two arguments:
The date string (e.g. "Mar 03, 2010")
The format string (e.g. "%b %d, %Y")
With just these two arguments, strptime() will return a datetime instance for March 3, 2010.
march3 = datetime.datetime.strptime("Mar 03, 2010", "%b %d, %Y")
"""

import datetime
mystery_date = datetime.datetime.strptime("11:00AM on Wednesday March 3, 2010", "%I:%M%p on %A %B %d, %Y")





# convert the Unix timestamp to a datetime object using datetime.datetime.fromtimestamp()

import datetime

for row in posts:
    # Convert the Unix timestamp, which is at index 2 of the row, to a floating point number.
    # Then convert the floating point number to a datetime instance
    row[2] = datetime.datetime.fromtimestamp(float(row[2]))





march_count = 0
for row in posts:
    # Check if the datetime instance at index 2 has a .month property equal to 3.
    if row[2].month == 3:
        march_count += 1




# Write a function that takes in an integer value and returns the number of posts that were made during that month.
def monthcount(month):
    month_count = 0
    for row in posts:
        if row[2].month == month:
            month_count += 1
    return month_count

feb_count = monthcount(2)
aug_count = monthcount(8)



