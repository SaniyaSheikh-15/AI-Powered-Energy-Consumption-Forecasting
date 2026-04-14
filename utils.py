import pandas as pd

def save_output(data):
    pd.DataFrame(data).to_csv("outputs/predictions.csv", index=False)
