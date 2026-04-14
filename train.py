import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.data_preprocessing import load_data, preprocess
from src.model import build_model

def train(data_path):
    df = load_data(data_path)
    df = preprocess(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = build_model()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    print("RMSE:", mean_squared_error(y_test,preds,squared=False))

    joblib.dump(model,"models/model.pkl")
    return model
