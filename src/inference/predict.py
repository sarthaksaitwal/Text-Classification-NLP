import numpy as np
from typing import List

from tensorflow.keras.models import load_model

from src.preprocessing.text_cleaning import TextCleaner
from src.tokenization.tokenizer import LSTMTokenizer
from src.utils.dataset_loader import AGNewsDatasetLoader


class NewsClassifier:
    """
    Inference class for LSTM-based news classification
    """

    def __init__(
        self,
        model_path: str = "saved_models/lstm_best_model.h5",
        train_path: str = "data/train.csv",
        test_path: str = "data/test.csv"
    ):
        # Load trained model
        self.model = load_model(model_path)

        # Label mapping
        self.label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }

        # Preprocessing (SAME as training)
        self.cleaner = TextCleaner(
            remove_stopwords=False,
            use_lemmatization=False
        )

        # Tokenizer (fit ONLY on training data)
        loader = AGNewsDatasetLoader(
            train_path=train_path,
            test_path=test_path
        )

        X_train, _, _, _ = loader.load_data()

        X_train_clean = self.cleaner.clean_corpus(X_train)

        self.tokenizer = LSTMTokenizer(
            vocab_size=20000,
            max_length=100
        )

        self.tokenizer.fit(X_train_clean)

    def predict(self, texts: List[str]):
        """
        Predict category for a list of texts
        """
        # Clean input text
        texts_clean = self.cleaner.clean_corpus(texts)

        # Tokenize & pad
        sequences = self.tokenizer.transform(texts_clean)

        # Predict probabilities
        probs = self.model.predict(sequences)

        predictions = []

        for prob in probs:
            class_id = int(np.argmax(prob))
            confidence = float(np.max(prob))

            predictions.append({
                "label": self.label_map[class_id],
                "confidence": round(confidence, 4)
            })

        return predictions

if __name__ == "__main__":
    classifier = NewsClassifier()

    samples = [
        "Apple releases a new AI-powered chip for laptops",
        "Manchester United wins the championship after dramatic final",
        "Global stock markets fall amid inflation concerns",
        "NASA announces new space telescope mission"
    ]

    results = classifier.predict(samples)

    for text, result in zip(samples, results):
        print(f"\nText: {text}")
        print(f"Prediction: {result['label']} ({result['confidence']})")
