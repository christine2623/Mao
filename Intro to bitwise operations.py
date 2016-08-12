# Demo
print(5 >> 4)  # Right Shift
print(5 << 1)  # Left Shift
print(8 & 5)   # Bitwise AND
print(9 | 4 )  # Bitwise OR
print(12 ^ 42) # Bitwise XOR
print(~88)     # Bitwise NOT


# In Python, you can write numbers in binary format by starting the number with 0b.
print(0b1,)   #1
print(0b10,)   #2
print(0b11,) #3
print(0b100,)  #4
print(0b101,)  #5
print(0b110,) #6
print(0b111)   #7
print("******")
print(0b1 + 0b11)
print(0b11 * 0b11)



# Count one to twelve in binary
one = 0b1
two = 0b10
three = 0b11
four = 0b100
five = 0b101
six =0b110
seven =0b111
eight =0b1000
nine =0b1001
ten =0b1010
eleven =0b1011
twelve = 0b1100


"""bin() takes an integer as input and returns the binary representation of that integer in a string.
(Keep in mind that after using the bin function, you can no longer operate on the value like a number.)"""
"""You can also represent numbers in base 8 and base 16 using the oct() and hex() functions."""
#print out bin(1) to bin(5)
print(bin(1))
print(bin(2))
print(bin(3))
print(bin(4))
print(bin(5))



"""int("110", 2)
# ==> 6
When given a string containing a number and the base that number is in,
the function will return the value of that number converted to base ten."""
print(int("1",2))
print(int("10",2))
print(int("111",2))
print(int("0b100",2))
print(int(bin(5),2))
# Print out the decimal equivalent of the binary 11001001.
print(int("11001001", 2))


#  left and right shift bitwise operators
"""Note that you can only do bitwise operations on an integer.
Trying to do them on strings or floats will result in nonsensical output!"""
shift_right = 0b1100
shift_left = 0b1

# Shift the shift_right variable right 2, (12 >> 2) means shift number 12 in binary rightwards for 2
shift_right = (12 >> 2)
shift_left = (1 << 2)

print(bin(shift_right))
# print out the number in binary
print(bin(shift_left))



"""The bitwise AND (&) operator compares two numbers on a bit level and returns a number
where the bits of that number are turned on if the corresponding bits of both numbers are 1. For example:

     a:   00101010   42
     b:   00001111   15
===================
 a & b:   00001010   10"""
print(bin(0b1110 & 0b101))



# The bitwise OR (|) operator
print(bin(0b1110 | 0b101))


"""The XOR (^) or exclusive or operator compares two numbers on a bit level and returns a number
where the bits of that number are turned on if either of the corresponding bits of the two numbers are 1, but not both.

    a:  00101010   42
    b:  00001111   15
================
a ^ b:  00100101   37"""
"""0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0"""
print(bin(0b1110 ^ 0b101))


# The bitwise NOT operator (~) just flips all of the bits in a single number.
# Mathematically, this is equivalent to adding one to the number and then making it negative.
print(~1)
print(~2)
print(~3)
print(~42)
print(~123)



#  We want to see if the fourth bit from the right is on.
def check_bit4(input):
    template = 0b1000
    compare = input & template
    if compare > 0:
        return "on"
    else:
        return "off"

print(check_bit4(0b1))
print(check_bit4(0b11011))
print(check_bit4(0b1010))


# Use masks to turn a bit in a number on using |
a = 0b10111011
mask = 0b100
result = a | mask
print(bin(result))



# XOR (^) operator is very useful for flipping bits
"""Using ^ on a bit with the number one will return a result where that bit is flipped."""
"""a = 0b110 # 6
mask = 0b111 # 7
desired =  a ^ mask # 0b1"""
a = 0b11101110
b = 0b11111111
result = a ^ b
print(bin(result))



#  Turn on the 10th bit from the right of the integer
"""a = 0b101
# Tenth bit mask
mask = (0b1 << 9)  # One less than ten
desired = a ^ mask"""


# Flip the nth bit (with the ones bit being the first bit) and store it in result
def flip_bit(numbers, n):
    mask = (0b1 << (n - 1))
    result = numbers ^ mask
    return bin(result)


print(flip_bit(0b10111010100101010, 7))