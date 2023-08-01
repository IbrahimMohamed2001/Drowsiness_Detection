from flask import Flask, render_template
import mysql.connector
import time

app = Flask(__name__)

@app.route("/")
def index():
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="radar"
    )
    cursor = connection.cursor()
    # Retrieve data from the database
    cursor.execute("SELECT * FROM alarm")
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template("index.html", rows=rows)

@app.route("/data")
def data():
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="radar"
    )
    cursor = connection.cursor()
    # Retrieve data from the database
    cursor.execute("SELECT * FROM alarm")
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return {"data": rows}

if __name__ == "__main__":
    app.run(debug=True,host="192.168.30.128")
