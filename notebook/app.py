from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form inputs
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        soil = float(request.form["soil"])
        crop = float(request.form["crop"])
        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        # Create input array
        features = np.array([[temperature,
                              humidity,
                              moisture,
                              soil,
                              crop,
                              nitrogen,
                              potassium,
                              phosphorous]])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Recommended Fertilizer: {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)