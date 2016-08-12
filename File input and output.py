# open()
"""f = open("output.txt", "w")
This told Python to open output.txt in "w" mode ("w" stands for "write").
We stored the result of this operation in a file object, f
Doing this opens the file in write-mode and prepares Python to send data into the file."""
my_list = [i ** 2 for i in range(1, 11)]

# "r+" as a second argument to the function so the file will allow you to read and write to it!
my_file = open("output.txt", "r+")
for value in my_list:
    # write() function takes a string argument
    # Make sure to add a newline ("\n") after each element to ensure each will appear on its own line
    my_file.write(str(value) + "\n")

# You must close the file. You do this simply by calling my_file.close()
# If you write to a file without closing, the data won't make it to the target file.
my_file.close()


# read(), read-only
my_file = open("output.txt", "r")

print(my_file.read())
my_file.close()



# readline() function read from a file line by line, rather than pulling the entire file in at once.
file = open("text.txt", "w")
file.write("I'm the first line of the file!\n")
file.write("I'm the second line.\n")
file.write("Third line here, boss.\n")
file.close()

my_file = open("text.txt", "r")
print(my_file.readline())
print(my_file.readline())
print(my_file.readline())
my_file.close()



""" with open("file", "mode") as variable:
    # Read or write to the file"""
# To close the file atomatically, add "with" and "as".

# Practice
with open("text.txt", "w") as my_file:
    my_file.write("Little Lion is so cute and hard-working!")



# Make sure it really closed
with open("text.txt", "w") as my_file:
    my_file.write("Little Lion is so cute and hard-working!")

# if my_file is open, the value of my_file.closed will be False
if my_file.closed == "False":
    my_file.close()

print(my_file.closed)