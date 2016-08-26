"""
 SQLite is a database that doesn't require a standalone server process and stores the entire database as a file on disk.
 This makes it ideal for working with larger datasets that can fit on disk but not in memory.
 Since the Pandas library loads the entire dataset we're working with into memory,
 this makes SQLite a compelling alternative option for working with datasets that are larger than 8 gigabytes (which is roughly the amount of memory modern computers contain).
 In addition, since the entire database can be contained within a single file, some datasets are released online as a SQLite database file (using the extension .db).
"""





# Connect To The Database
# Import the Sqlite3 library into the environment.
import sqlite3
"""
Once imported, we connect to the database we want to query using the connect() function.
The connect() function has a single required parameter, the database we want to connect to.
Since the database we're working with is stored as a file on disk, we need to pass in the filename.
The connect() function returns a Connection instance, which maintains the connection to the database we want to work with.
When you're connected to a database, SQLite locks the database file and prevents any other process from connecting to the database simultaneously.
This was a design decision made by the SQLite team to keep the database lightweight and avoid the complexity that arises
when multiple processes are interacting with the same database.
"""
# use the Sqlite3 function connect() to connect to jobs.db and assign the returned Connection instance to conn.
conn = sqlite3.connect("jobs.db")






# Cursor Object And Tuples
"""
Before we can execute a query, we need to express our SQL query as a string.
While we use the Connection class to represent the database we're working with, we use the Cursor class to:

run a query against the database.
parse the results from the database.
convert the results to native Python objects.
store the results within the Cursor instance as a local variable.

After running a query and converting the results to a list of tuples, the Cursor instance stores the list as a local variable.
Before diving into the syntax of querying the database, let's learn more about tuples.
"""




# Tuples
"""
A tuple is a core Python data structure used to represent a sequence of values, similar to a list.
Unlike lists, tuples are immutable, which means they can't be modified after creation.
Each row is in the results set is represented as a tuple.

To create an empty tuple, assign a pair of empty parentheses to a variable:
t = ()

Tuples are indexed the same way as lists, from 0 to n-1, and you access values using bracket notation.

t = ('Apple', 'Banana')
apple = t[0]
banana = t[1]

Tuples are faster than lists, which is helpful when working with larger databases and larger results sets.
"""




# Running A Query
"""
We need to use the Connection instance method cursor() to return a Cursor instance corresponding to the database we want to query.
cursor = conn.cursor()
"""
"""Example:
# SQL Query as a string
query = "select * from recent_grads;"
# Execute the query, convert the results to tuples, and store as a local variable.
cursor.execute(query)
# Fetch the full results set, as a list of tuples.
results = cursor.fetchall()
# Display the first 3 results.
print(results[0:3])
"""
# Write a query that returns all of the values in the Major column from the recent_grads table.
cursor = conn.cursor()

# Create a table
import pandas
recent_grads = pandas.read_csv("recent-grads.csv")
recent_grads = recent_grads.iloc[:,:21]

# Extract all the column names into a string
recent_grads_column = recent_grads.columns.tolist()
print(recent_grads_column)
total_column_name = ""
for index in range(0, len(recent_grads_column)):
    if index == 0:
        total_column_name = recent_grads_column[index]
    else:
        total_column_name = total_column_name + "," + recent_grads_column[index]
# total_column_name = total_column_name.replace(",Unnamed: 21", "")
print(total_column_name)

# Create table XXX (string)
# Table only can be created once!!! No re-run
# cursor.execute("CREATE TABLE recent_grads (" + total_column_name + ");")

# Insert rows of data
# Write the dataframe into sqlite
for index in range(0, recent_grads.shape[0]):
    # print(index)
    recent_grads_values = recent_grads.iloc[index].values.tolist()
    # Insert into XXX values (string, string, string)
    row_value = ""
    for index in range(0, len(recent_grads_values)):
        if index == 0:
            row_value = "'" + str(recent_grads_values[index]) + "'"
        else:
            row_value = row_value + ", " + "'" +str(recent_grads_values[index])+ "'"
    cursor.execute("INSERT INTO recent_grads VALUES (" + row_value + ");")
# Another way: recent_grads.to_sql("recent_grads", conn, if_exists='append', index=False)
# df.to_sql("table_name", connection, if_exists, index)

# Save (commit) the changes
# Insert needs commit !!
# connection.commit()
# conn.commit()



query = "select * from recent_grads;"
cursor.execute(query)
# Store the full results set (a list of tuples) in majors.
total = cursor.fetchall()
# Then, print the first value of the first tuple in majors.
print(total[0][0])

query = "select Major from recent_grads;"
cursor.execute(query)
# Store the full results set (a list of tuples) in majors.
majors = cursor.fetchall()
# Then, print the first 3 tuples in majors.
print(majors[:3])




# Shortcut For Running A Query
# The sqlite library actually allows us to skip creating a Cursor altogether by using the execute method within the Connection object itself.
"""Example:
conn = sqlite3.connect("jobs.db")
query = "select * from recent_grads;"
conn.execute(query).fetchall()
"""



# Fetching A Specific Number Of Results
# To return a single result (as a tuple), we use the Cursor method fetchone() and to return n results, we use the Cursor method fetchmany().
"""
Each Cursor instance contains an internal counter which is updated every time you retrieve results.
When you call the fetchone() method, the Cursor instance will return a single result and then increment its internal counter by 1.
This means that if you call fetchone() again, the Cursor instance will actually return the second tuple in the results set (and increment by 1 again).
"""
"""
The fetchmany() method takes in an integer (n) and returns the corresponding results starting from the current position.
The fetchmany() method then increments the Cursor instance's counter by n.
In the following code, we return the first 2 results using the fetchone() method, then the next 5 results using the fetchmany() method.
"""
"""Example:
first_result = cursor.fetchone()
second_result = cursor.fetchone()
next_five_results = cursor.fetchmany(5)
"""
# Write and run a query that returns the Major and Major_category columns from recent_grads.
query = "select major, major_category from recent_grads;"
cursor.execute(query)
# Then, fetch the first 5 results and store it as five_results.
five_results = cursor.fetchmany(5)
print(five_results)



# Closing The Connection
"""
Since SQLite restricts access to the database file when we're connected to a database,
we need to close the connection when we're done working with it.

In addition, if we made any changes to the database, they are automatically saved and our changes are persisted in the database file upon closing.
"""




# Practice
# Write and execute a query that returns all of the major names (Major) in reverse alphabetical order (Z to A).
query = "select major from recent_grads order by major DESC;"
cursor.execute(query)
# Assign the full result set to reverse_alphabetical.
reverse_alphabetical = cursor.fetchall()
# Finally, close the connection to the database.
print(reverse_alphabetical)



# pandas.read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)¶
# Read SQL query into a DataFrame.
query= "select * from recent_grads"
recent_grads_2 = pandas.read_sql_query(query, conn)
recent_grads_2 = recent_grads_2.dropna(axis=0)




# Do not forget to close the db
conn.close()