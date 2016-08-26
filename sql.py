# https://docs.python.org/3.5/library/sqlite3.html

import sqlite3
sqliteConnection = sqlite3.connect('example.db')
sqliteCursor = sqliteConnection.cursor()

# Create table
# sqliteCursor.execute("CREATE TABLE stocks (date text, trans text, symbol text, qty real, price real)")
#
#
import pandas
inputCsv = pandas.read_csv("inputCsv.csv")
input_csv_column = inputCsv.columns.tolist()
# inputCsv_columnName.split(",")
total_column_name = ""
for index in range(0, len(input_csv_column)):
    if index == 0:
        total_column_name = input_csv_column[index]
    else:
        total_column_name = total_column_name + "," + input_csv_column[index]
    print(total_column_name)

sqliteCursor.execute("CREATE TABLE inputCsv_take6 (" + total_column_name + ")")
# Insert a row of data
sqliteCursor.execute("INSERT INTO stocks VALUES ('2006-01-06','BUY','RHAT',100,35.14)")

# Save (commit) the changes
# Insert needs commit !!
sqliteConnection.commit()

sqliteCursor.execute("SELECT * FROM stocks")
print(sqliteCursor.fetchone())
print(sqliteCursor.fetchall())

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
sqliteConnection.close()