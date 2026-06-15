from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    pclass = int(request.form["Pclass"])
    sex = int(request.form["Sex"])
    age = float(request.form["Age"])
    fare = float(request.form["Fare"])

    data = np.array([[pclass, sex, age, fare]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Passenger Survived ✅"
    else:
        result = "Passenger Not Survived ❌"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)