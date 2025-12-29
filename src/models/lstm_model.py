from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

def build_lstm_model(
    vocab_size:int,
    max_length:int,
    embedding_dim:int=128,
    lstm_units:int=128,
    num_classes:int=4
):
    """
    Builds and returns an LSTM-based text classification model
    """

    model=Sequential()

    # Explicit input layer (IMPORTANT)
    model.add(Input(shape=(max_length,)))

    # Embedding layer
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            # input_length=max_length
        )
    )

    # LSTM layer
    model.add(
        Bidirectional(LSTM(
            lstm_units,
            return_sequences=False
        ))
    )

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model