import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

from src.utils.dataset_loader import AGNewsDatasetLoader
from src.preprocessing.text_cleaning import TextCleaner
from src.tokenization.tokenizer import LSTMTokenizer


def evaluate_lstm_model():
    # -----------------------------
    # Paths
    # -----------------------------
    MODEL_PATH = "saved_models/lstm_best_model.h5"  # or .keras
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test.csv"

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    loader = AGNewsDatasetLoader(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH
    )
    X_train, y_train, X_test, y_test = loader.load_data()

    # -----------------------------
    # 2. Preprocess text (SAME as training)
    # -----------------------------
    cleaner = TextCleaner(
        remove_stopwords=False,
        use_lemmatization=False
    )

    X_train_clean = cleaner.clean_corpus(X_train)
    X_test_clean = cleaner.clean_corpus(X_test)

    # -----------------------------
    # 3. Tokenization (fit on TRAIN only)
    # -----------------------------
    tokenizer = LSTMTokenizer(
        vocab_size=20000,
        max_length=100
    )

    tokenizer.fit(X_train_clean)
    X_test_pad = tokenizer.transform(X_test_clean)

    # -----------------------------
    # 4. Load trained model
    # -----------------------------
    model = load_model(MODEL_PATH)

    # -----------------------------
    # 5. Predict
    # -----------------------------
    y_pred_probs = model.predict(X_test_pad, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # -----------------------------
    # 6. Metrics
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    target_names = ["World", "Sports", "Business", "Sci/Tech"]

    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names
        )
    )

    # -----------------------------
    # 7. Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("LSTM Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate_lstm_model()
