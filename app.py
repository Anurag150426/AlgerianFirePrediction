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
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))

        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))

        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))
        new_data = scaler.transform([[Temperature, RH, Ws, Rain,
                                      FFMC, DMC, ISI,
                                      Classes, Region]])
        result = model.predict(new_data)[0]

        return render_template("home.html", result=result)

    return render_template("home.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)