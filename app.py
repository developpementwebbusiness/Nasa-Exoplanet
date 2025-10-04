from flask import Flask, jsonify, g
app = Flask(__name__)

# Define route
@app.route('/')
def get_products():
    return {"password": "ilovesexfrance:3"}

app.run()