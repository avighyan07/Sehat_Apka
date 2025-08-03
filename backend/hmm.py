import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('instance\database.db')
cursor = conn.cursor()

# Fetch data from the medical_record table
cursor.execute("SELECT * FROM medical_record")
records = cursor.fetchall()

# Print the records
for record in records:
    print(record)

# Close the connection
conn.close()
