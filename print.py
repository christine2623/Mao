


print("test\\n")
print("test\n")
print("test")
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\
      bbbbbbbbbbbbbbbbbbbbbbbbbbb")


f = open('output.txt', 'r')
data = f.read()
rows = data.split('\n')
print(rows)

five_elements = [['Albuquerque', '749'], ['Anaheim', '371'], ['Anchorage', '828'], ['Arlington', '503'], ['Atlanta', '1379']]
['Albuquerque', 'Anaheim', 'Anchorage', 'Arlington', 'Atlanta']
print(five_elements)
cities_list = []
for cityList in five_elements:
    cities_list.append(cityList[0])
print(cities_list)


rows = ['Albuquerque, 749', 'Anaheim, 371', 'Anchorage, 828', 'Arlington, 503', 'Atlanta, 1379']
print(rows)
print(rows[0:5])
int_crime_rates = []
for stringElement in rows:
    listElement = stringElement.split(", ")
    int_crime_rates.append(int(listElement[1]))
print(int_crime_rates)