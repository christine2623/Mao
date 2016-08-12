# member variables are information that belongs to the class object.
# We use dot notation to access the member variables of classes since those variables belong to the object.
class Car(object):
    condition = "new"

    def __init__(self, model, color, mpg):
        self.model = model
        self.color = color
        self.mpg = mpg

    def display_car(self):
        return "This is a %s %s with %s MPG." % (self.color, self.model, str(self.mpg))

    def drive_car(self):
        self.condition = "used"


class ElectricCar(Car):
    def __init__(self, model, color, mpg, battery_type):
        self.model = model
        self.color = color
        self.mpg = mpg
        self.battery_type = battery_type

    def drive_car(self):
        self.condition = "like new"


my_car = ElectricCar("Tesla", "black", 100, "molten salt")
my_car2 = Car("DeLorean", "silver", 88)
print(my_car.display_car())
print(my_car.condition)
print(my_car.drive_car())
print(my_car.condition)



#__repr__(self)
class Point3D(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # tells Python to represent this object in the following format: (x, y, z)
    def __repr__(self):
        return "(%d, %d, %d)" % (self.x, self.y, self.z)


my_point = Point3D(1, 2, 3)
print(my_point)
