from src.train import train
from src.predict import predict
from src.utils import save_output

if __name__ == "__main__":
    train("data/sample.csv")

    sample = {"feature1":10,"feature2":20,"feature3":30}
    pred = predict(sample)

    print("Prediction:",pred)
    save_output([{"prediction":pred}])
