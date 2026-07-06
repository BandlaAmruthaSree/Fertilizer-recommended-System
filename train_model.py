import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset/fertilizer.csv")

le = LabelEncoder()

df["Soil Type"] = le.fit_transform(df["Soil Type"])
df["Crop Type"] = le.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = le.fit_transform(df["Fertilizer Name"])

X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("Model Saved Successfully")