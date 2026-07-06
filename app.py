from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Fertilizer Recommendation System is Running Successfully!"

if __name__ == "__main__":
    app.run(debug=True)
    from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    moisture = float(request.form["moisture"])
    soil = float(request.form["soil"])
    crop = float(request.form["crop"])
    nitrogen = float(request.form["nitrogen"])
    potassium = float(request.form["potassium"])
    phosphorous = float(request.form["phosphorous"])

    data = np.array([[temperature,
                      humidity,
                      moisture,
                      soil,
                      crop,
                      nitrogen,
                      potassium,
                      phosphorous]])

    prediction = model.predict(data)

    fertilizer = encoder.inverse_transform(prediction)

    return render_template("result.html",
                           prediction=fertilizer[0])


if __name__ == "__main__":
    app.run(debug=True)