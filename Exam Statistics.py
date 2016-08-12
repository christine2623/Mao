grades = [100, 100, 90, 40, 80, 100, 85, 70, 90, 65, 90, 85, 50.5]
print("Grades:", grades)


# def a function to print each grade at a time.
def print_grades(grades):
    for i in grades:
        print(i)



# python built-in sum() function
sum_grades = sum(grades)



# Computing the sum manually involves computing a rolling sum
def grades_sum(scores):
    total = 0
    for i in scores:
        total += i
    return total





# perform average
def grades_average(grades):
    avg = grades_sum(grades) / float(len(grades))
    return avg


# perform variance
def grades_variance(scores):
    average = grades_average(scores)
    variance = 0
    for i in scores:
        variance += (average - i) ** 2
    result = variance / float(len(scores))
    return result




# standard deviation
def grades_std_deviation(variance):
    # square root can be wrote as "** 0.5"
    return variance ** 0.5

variance = grades_variance(grades)


#print out all the stats
print("Grades:", print_grades(grades))
print("Sum:", grades_sum(grades))
print("Average:", grades_average(grades))
print("Variance:", grades_variance(grades))
print("Standard Deviation:", grades_std_deviation(variance))