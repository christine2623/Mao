# Class Syntax
""" A basic class consists only of the class keyword, the name of the class, and the class from which the new class inherits in parentheses."""
"""class NewClass(object):
    # Class magic here"""

# create a class
class Animal(object):
    pass
"""pass doesn't do anything, but it's useful as a placeholder in areas of your code where Python expects an expression."""


"""__init__() function is required for classes, and it's used to initialize the objects it creates.
__init__() always takes at least one argument, self, that refers to the object being created."""
"""Python will use the first parameter that __init__() receives to refer to the object being created"""
# using __init__()
class Animal(object):
    def __init__(self, name):
        # let the function know that name refers to the created object's name
        self.name = name

zebra = Animal("Jeffrey")

print(zebra.name)


# More for __init__()
# Class definition
class Animal(object):
    """Makes cute animals."""
    # For initializing our instance objects
    def __init__(self, name, age, is_hungry):
        self.name = name
        self.age = age
        self.is_hungry = is_hungry

# Note that self is only used in the __init__()
# function definition; we don't need to pass it
# to our instance objects.

zebra = Animal("Jeffrey", 2, True)
giraffe = Animal("Bruce", 1, False)
panda = Animal("Chad", 7, True)

print(zebra.name, zebra.age, zebra.is_hungry)
print(giraffe.name, giraffe.age, giraffe.is_hungry)
print(panda.name, panda.age, panda.is_hungry)


"""not all variables are accessible to all parts of a Python program at all times.
When dealing with classes, you can have variables that are available everywhere (global variables),
variables that are only available to members of a certain class (member variables),
and variables that are only available to particular instances of a class (instance variables)."""

class Animal(object):
    """Makes cute animals."""
    # is_alive is accessible to all members of the Animal class
    is_alive = True
    def __init__(self, name, age):
        self.name = name
        self.age = age

zebra = Animal("Jeffrey", 2)
giraffe = Animal("Bruce", 1)
panda = Animal("Chad", 7)

print(zebra.name, zebra.age, zebra.is_alive)
print(giraffe.name, giraffe.age, giraffe.is_alive)
print(panda.name, panda.age, panda.is_alive)



"""When a class has its own functions, those functions are called methods.
You've already seen one such method: __init__()"""

# create description function
class Animal(object):
    """Makes cute animals."""
    is_alive = True
    health = "good"
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def description(self):
        print(self.name)
        print(self.age)

hippo = Animal("Leo", "18")
print(hippo.description())

sloth = Animal("Leo", "27")
ocelot = Animal("Leonard", "40")
print(hippo.health)
print(sloth.health)
print(ocelot.health)



# Real world application on Shopping cart
class ShoppingCart(object):
    """Creates shopping cart objects for users of our fine website."""
    items_in_cart = {}
    def __init__(self, customer_name):
        self.customer_name = customer_name

    def add_item(self, product, price):
        """Add product to the cart."""
        if not product in self.items_in_cart:
            self.items_in_cart[product] = price
            print(product + " added.")
        else:
            print(product + " is already in the cart.")

    def remove_item(self, product):
        """Remove product from the cart."""
        if product in self.items_in_cart:
            del self.items_in_cart[product]
            print(product + " removed.")
        else:
            print(product + " is not in the cart.")


my_cart = ShoppingCart("Christine")
print(my_cart.add_item("Fatty_dog", "10000000"))


# Inheritance = "is a " relationship
# a Panda is a bear, so a Panda class could inherit from a Bear class.
"""We've defined a class, Customer, as well as a ReturningCustomer class that inherits from Customer.
Note that we don't define the display_cart method in the body of ReturningCustomer,
but it will still have access to that method via inheritance. """
class Customer(object):
    """Produces objects that represent customers."""
    def __init__(self, customer_id):
        self.customer_id = customer_id

    def display_cart(self):
        print("I'm a string that stands in for the contents of your shopping cart!")

class ReturningCustomer(Customer):
    """For customers of the repeat variety."""
    def display_order_history(self):
        print("I'm a string that stands in for your order history!")

monty_python = ReturningCustomer("ID: 12345")
monty_python.display_cart()
monty_python.display_order_history()



""" inheritance works like this:
class DerivedClass(BaseClass):
    # code goes here"""
# Inheritance practice
class Shape(object):
    """Makes shapes!"""
    def __init__(self, number_of_sides):
        self.number_of_sides = number_of_sides

class Triangle(Shape):
    def __init__(self, side1, side2, side3):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3


# Override
class Employee(object):
    """Models real-life employees!"""
    def __init__(self, employee_name):
        self.employee_name = employee_name

    def calculate_wage(self, hours):
        self.hours = hours
        return hours * 20.00

class PartTimeEmployee(Employee):
    # models part-time employee
    def calculate_wage(self, hours):
        """PartTimeEmployee.calculate_wage overrides Employee.calculate_wage, it still needs to set self.hours = hours"""
        self.hours = hours
        return hours * 12.00



# You override the method in the derived class while you want to use the method defined in the base class.
class Employee(object):
    """Models real-life employees!"""

    def __init__(self, employee_name):
        self.employee_name = employee_name

    def calculate_wage(self, hours):
        self.hours = hours
        return hours * 20.00


class PartTimeEmployee(Employee):
    # models part-time employee
    def calculate_wage(self, hours):
        """PartTimeEmployee.calculate_wage overrides Employee.calculate_wage, it still needs to set self.hours = hours"""
        self.hours = hours
        return hours * 12.00

    def full_time_wage(self, hours):
        # Call a super cell to eruse the method defined in the base class
        return super(PartTimeEmployee, self).calculate_wage(hours)


milton = PartTimeEmployee("Fatty_dog")
print(milton.full_time_wage(10))


# Practice
class Triangle(object):
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
    number_of_sides = 3
    def check_angles(self):
        if self.angle1 + self.angle2 + self.angle3 == 180:
            return True
        else:
            return False

class Equilateral(Triangle):
    angle = 60
    def __init__(self):
        # Inheritance self.angle1 for the Triangle class and overrite it
        self.angle1 = self.angle
        self.angle2 = self.angle
        self.angle3 = self.angle

my_triangle = Triangle(90, 60, 30)
print(my_triangle.number_of_sides)
print(my_triangle.check_angles())