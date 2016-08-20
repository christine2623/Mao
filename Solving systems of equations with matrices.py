# Systems Of Equations As Matrices
# The simplest way to represent a matrix in python is a numpy array. A numpy array can have rows and columns, just like a matrix.
import numpy as np

# Set the dtype to float to do float math with the numbers.
matrix = np.asarray([[2, 1, 25],[3, 2, 40]], dtype=np.float32)
# Multiply the first row of the matrix by two.
matrix[0] *= 2
# Then subtract the second row from the first row.
matrix[0] -= matrix[1]
# Then, subtract three times the first row from the second row.
matrix[1] -= (3*matrix[0])
# Finally, divide the second row by 2 to get rid of the coefficient.
matrix[1] /= 2
# At the end, the first row should indicate that x equals 10, and the second row should indicate that y equals 5. We just solved our equation with matrices!
print(matrix)




# Gauss's Method
"""
If a linear system is changed to another by one of the following operations:

(1) an equation is swapped with another
(2) an equation has both sides multiplied by a nonzero constant
(3) an equation is replaced by the sum of itself and a multiple of another

then the two systems have the same set of solutions.
"""




# Solving More Complex Equations
matrix = np.asarray([
    [1, 2, 0, 7],
    [0, 3, 3, 11],
    [1, 2, 2, 11]
], dtype=np.float32)
matrix[2] -= matrix[0]
matrix[2] /= 2
matrix[1] -= (3*matrix[2])
matrix[1] /= 3
matrix[0] -= (2*matrix[1])

print(matrix)




# Echelon Form
# Echelon form is what happens when the leading variable of each row is to the right of the leading variable in the previous row.
# Any rows consisting of all zeros should be at the bottom.
# This is where row swapping can come in handy
matrix = np.asarray([
    [0, 0, 0, 7],
    [0, 0, 1, 11],
    [1, 2, 2, 11],
    [0, 5, 5, 1]
], dtype=np.float32)

# Swap the first and the third rows - first swap
matrix[[0,2]] = matrix[[2,0]]
matrix[[1,3]] = matrix[[3,1]]
matrix[[2,3]] = matrix[[3,2]]

print(matrix)





# Reduced Row Echelon Form
"""
Generally, the way to solve systems of linear equations is to first try to get them into reduced row echelon form.
We just covered echelon form.
Reduced row echelon form meets all the same conditions as echelon form, except every leading variable must equal 1,
and it must be the only nonzero entry in its column.
"""
"""
Generally, to get to reduced row echelon form, we repeat the following steps:

1. Start on a new row
2. Perform any needed swaps to move the leftmost available leading coefficient to the current row
3. Divide the row by its leading coefficient to make the leading coefficient equal 1
4. Subtract the row from all other rows (with an appropriate multiplier) to ensure that its leading variable
is the only nonzero value in its column.
5. Repeat until entire matrix is in reduced row-echelon form.
"""
A = np.asarray([
        [0, 2, 1, 5],
        [3, 0, 1, 10],
        [1, 2, 1, 8]
        ], dtype=np.float32)

# First, we'll swap the second row with the first to get a nonzero coefficient in the first column
A[[0,1]] = A[[1,0]]

# Then, we divide the first row by 3 to get a coefficient of 1
A[0] /= 3

# Now, we need to make sure that our 1 coefficient is the only coefficient in its column
# We have to subtract the first row from the third row
A[2] -= A[0]

# Now, we move to row 2
# We divide by 2 to get a one as the leading coefficient
A[1] /= 2

# We subtract 2 times the second row from the third to get rid of the second column coefficient in the third row
A[2] -= (2 * A[1])

# Now, we can move to the third row, but it already looks good.
# We're finished, and our system is solved!
print(A)





# Inconsistency
# Not all systems of equations can be solved. In the cases where they can't, we call the system inconsistent.
# An inconsiste system will have two or more equations that conflict, making it impossible to find a solution.

# Find whether A is consistent by attempting to convert it to reduced row echelon form.
# Assign True to A_consistent if it is, False if it isn't.
A = np.asarray([
    [10, 5, 20, 60],
    [3, 1, 0, 11],
    [8, 2, 2, 30],
    [0, 4, 5, 13]
], dtype=np.float32)
A[0] /= 10
A[1] -= (3*A[0])
A[2] -= (8*A[0])
A[[3,1]] = A[[1,3]]
A[1] /= 4
A[0] *= 2
A[0] -= A[1]
A[0] /= 2
A[2] /= 2
A[2] += A[1]
A[3] *= 2
A[3] += A[1]
A_consistent = True

# Find whether B is consistent by attempting to convert it to reduced row echelon form.
# Assign True to B_consistent if it is, False if it isn't.
B = np.asarray([
    [5, -1, 3, 14],
    [0, 1, 2, 8],
    [0, -2, 5, 1],
    [0, 0, 6, 6]
], dtype=np.float32)
B[0] /= 5
B[2] += (B[1]*2)
B_consistent = False




# Infinite Solutions
# Check whether A has infinite solutions.
# If it does, assign True to A_infinite.
# If it doesn't, False to A_infinite.
A = np.asarray([
        [2, 4, 8, 20],
        [4, 8, 16, 40],
        [20, 5, 5, 10]
], dtype=np.float32)
A[[1,2]] = A[[2,1]]
A[1] /= 5
A[0] /= 2
A[2] /= 4
A_infinite = True

# Check whether B has infinite solutions.
# If it does, assign True to B_infinite
# If it doesn't, assign False to B_infinite
B = np.asarray([
        [1, 1, 1, 4],
        [3, -2, 5, 8],
        [8, -4, 5, 10]
        ], dtype=np.float32)
B[1] -= (3*B[0])
B[2] -= (8*B[0])
B[0] += B[1]
B[1] /= (-5)
B[2] -= (12*B[1])
B[2] *= (5/9)
B[1] *= (-1)
B[0] -= ((3/5)*B[2])
B[1] -= ((2/5)*B[2])
B_infinite = False





# Homogeneity
# A linear equation is homogeneous if it has a constant of zero.
# A system of equations that is homogeneous always has at least one solution --
# setting each variable to zero will always solve the system.




# Singularity
# A matrix is square if it has the same number of columns as rows
# A square matrix is singular if it represents a homogeneous system with infinite solutions.
