# sum up 6 and 11
total = sum([6,11])

"""total = 15
print(add(10, 20))
print(total)
"""
"""Since functions are designed to be reusable, they have to be isolated from the rest of the program.
Even though there's a variable called total inside the add function,
that variable is not connected to the total variable in our script.
The above script would first print out 30, then print out 15.
This is because the variable total defined in our script is in the global scope,
whereas the total variable inside add is in a local scope."""

"""The global scope is whatever happens outside of a function. Anything that happens in a function is a local scope.
There's only one global scope, but each function creates its own local scope."""


"""When you create a global variable, you can't create it and assign a value to it on the same line.
You first define the variable as global using the global keyword, then assign a value to it on a separate line."""

def test_function():
    # a was defined with the global keyword.
    global a
    a = 10

test_function()
print(a)


"""When you use a variable anywhere in a Python script, the Python interpreter will look for the value of that variable using some simple rules:

First start with the local scope, if any. If the variable is defined here, use the value.
Look at any enclosing scopes, starting with the innermost. These are "outside" local scopes. If the variable is defined in any of them, use the value.
Look in the global scope. If the variable is there, use the value.
Finally, look in the built-in functions.
If no value is found, an error will be thrown.
A simple way to remember this is LEGBE, which stands for "Local, Enclosing, Global, Built-ins, Error"."""

def find_total(l):
    total = sum(l)
    return total

def find_length(l):
    length = len(l)
    return length

def find_average(l):
    total = 10
    return find_total(l) / find_length(l)

print(find_average([1,10]))

total = 15
mode = 1
def find_total2(l, mode):
    # bad coding style
    # if mode == 1 then do mode 1 thing
    # else if mode == 2 then do mode 2 thing

    # bad coding style
    return total

def find_length2(l):
    length = len(l)
    return length

def find_average2(l, mode):
    total = 0 # bad coding style
    return find_total2(l, mode) / find_length2(l)

principal_outstanding_240 = [1606, 1567, 269, 184, 93, 75, 52, 88, 12, 5]

# bad coding style
mode = 1
find_average2(principal_outstanding_240, 1)

# bad coding style
total = 20
mode = 2
default_average = find_average2(principal_outstanding_240, 2)
