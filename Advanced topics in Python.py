# Anonymous Functions
my_list = range(16)
print(list(filter(lambda x: x % 3 == 0, my_list)))

# Print "Python"
languages = ["HTML", "JavaScript", "Python", "Ruby"]
print(list(filter(lambda x: x == languages[2], languages)))


# print out only the squares that are between 30 and 70
squares = [i**2 for i in range(1,11)]
print(list(filter(lambda x: 30 < x < 70, squares)))


# print out all the items (hint, hint) in the dictionary
movies = {
	"Monty Python and the Holy Grail": "Great",
	"Monty Python's Life of Brian": "Good",
	"Monty Python's Meaning of Life": "Okay"
}

print( movies.items())


# list of numbers between 1 and 15 (inclusive) that are evenly divisible by 3 or 5.
threes_and_fives = [i for i in range(1,16) if i % 3 == 0 or i % 5 == 0]


# Reverse and get the other charcs
garbled = "!XeXgXaXsXsXeXmX XtXeXrXcXeXsX XeXhXtX XmXaX XI"
message = garbled[::-2]
print(message)


# Getting the not X charcs in the sring
garbled = "!XeXgXaXsXsXeXmX XtXeXrXcXeXsX XeXhXtX XmXaX XI"

new_list = list(reversed(garbled))
a_list = ""
for index in range(0, len(new_list)):
    if new_list[index] != "X":
        a_list += new_list[index]

print(a_list)


# Getting the not X charcs in the sring
garbled = "!XeXgXaXsXsXeXmX XtXeXrXcXeXsX XeXhXtX XmXaX XI"

new_list = list(reversed(garbled))
a_list = ""
for index in range(0, len(new_list)):
    if index % 2 == 0:
        a_list += new_list[index]

print(a_list)



# Exclude the "X"
garbled = "IXXX aXXmX aXXXnXoXXXXXtXhXeXXXXrX sXXXXeXcXXXrXeXt mXXeXsXXXsXaXXXXXXgXeX!XX"
message = list(filter(lambda x: x != "X", garbled))
messages = "".join(message)
print(messages)
