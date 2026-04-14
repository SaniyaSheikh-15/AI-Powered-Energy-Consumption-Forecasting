import joblib
import pandas as pd

def load_model():
    return joblib.load("models/model.pkl")

def predict(data):
    model = load_model()
    df = pd.DataFrame([data])
    return model.predict(df)[0]
