import pandas as pd
from typing import List, Tuple


class AGNewsDatasetLoader:
    """
    Loads pre-split AG News train and test CSV files
    """

    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Returns:
            X_train, y_train, X_test, y_test
        """

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Combine title and description
        X_train = (train_df["Title"] + " " + train_df["Description"]).tolist()
        X_test = (test_df["Title"] + " " + test_df["Description"]).tolist()

        # Convert labels from 1–4 to 0–3
        y_train = (train_df["Class Index"] - 1).tolist()
        y_test = (test_df["Class Index"] - 1).tolist()

        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    loader = AGNewsDatasetLoader(
        train_path="data/train.csv",
        test_path="data/test.csv"
    )

    X_train, y_train, X_test, y_test = loader.load_data()

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("Sample text:", X_train[0])
    print("Sample label:", y_train[0])
