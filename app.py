import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('models/ridge_model.pkl','rb'))
scaler = pickle.load(open("models/scaler.pkl","rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get("Temperature", 0))
            RH = float(request.form.get("RH", 0))
            Ws = float(request.form.get("Ws", 0))
            Rain = float(request.form.get("Rain", 0))
            FFMC = float(request.form.get("FFMC", 0))
            DMC = float(request.form.get("DMC", 0))
            ISI = float(request.form.get("ISI", 0))
            Classes = float(request.form.get("Classes", 0))
            Region = float(request.form.get("Region", 0))

            new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = model.predict(new_data)[0]

            return render_template("home.html", result=round(result, 2))
        
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)