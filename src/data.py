import pandas as pd
from sklearn.model_selection import train_test_split

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def load_data(url=URL):
    return pd.read_csv(url, sep=";")


def prepare_data(df):
    y = df["quality"]
    x = df.drop(columns=["quality"])
    return x, y


def split_data(x, y, test_size=0.25, random_state=123456):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
