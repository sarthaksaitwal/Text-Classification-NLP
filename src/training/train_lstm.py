import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.dataset_loader import AGNewsDatasetLoader
from src.preprocessing.text_cleaning import TextCleaner
from src.tokenization.tokenizer import LSTMTokenizer
from src.models.lstm_model import build_lstm_model


def train_lstm():
    # -----------------------------
    # Paths
    # -----------------------------
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test.csv"
    MODEL_DIR = "saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    loader = AGNewsDatasetLoader(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH
    )

    X_train, y_train, X_test, y_test = loader.load_data()

    print("Dataset loaded")
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    # -----------------------------
    # 2. Preprocess text
    # -----------------------------
    cleaner = TextCleaner(
        remove_stopwords=False,
        use_lemmatization=False
    )

    X_train_clean = cleaner.clean_corpus(X_train)
    X_test_clean = cleaner.clean_corpus(X_test)

    print("Text preprocessing completed")

    # -----------------------------
    # 3. Tokenization & Padding
    # -----------------------------
    VOCAB_SIZE = 20000
    MAX_LENGTH = 100

    tokenizer = LSTMTokenizer(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH
    )

    X_train_pad = tokenizer.fit_transform(X_train_clean)
    X_test_pad = tokenizer.transform(X_test_clean)

    X_train_pad = np.array(X_train_pad)
    X_test_pad = np.array(X_test_pad)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    actual_vocab_size = tokenizer.get_vocab_size()
    print("Vocabulary size:", actual_vocab_size)

    # -----------------------------
    # 4. Build LSTM model
    # -----------------------------
    model = build_lstm_model(
        vocab_size=actual_vocab_size,
        max_length=MAX_LENGTH
    )

    model.summary()

    # -----------------------------
    # 5. Callbacks
    # -----------------------------
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    checkpoint_path = os.path.join(MODEL_DIR, "lstm_best_model.h5")
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )

    # -----------------------------
    # 6. Train model
    # -----------------------------
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=15,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    print("Training completed")
    print(f"Best model saved at: {checkpoint_path}")


if __name__ == "__main__":
    train_lstm()
