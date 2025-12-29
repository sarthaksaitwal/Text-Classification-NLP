from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from typing import List

class LSTMTokenizer:
    """
    Tokenizer & padding utility for LSTM models
    """
    def __init__(
            self,
            vocab_size:int=20000,
            max_length:int=200,
            oov_token:str="<OOV>"
        ):

        self.vocab_size=vocab_size
        self.max_length=max_length
        self.oov_token=oov_token

        self.tokenizer=Tokenizer(
            num_words=self.vocab_size,
            oov_token=self.oov_token
        )
    
    def fit(self,texts:List[str]):
        """
        Fit tokenizer on training texts
        """
        self.tokenizer.fit_on_texts(texts)
    
    def transform(self,texts:List[str]):
        """
        Convert texts to padded sequences
        """

        sequences=self.tokenizer.texts_to_sequences(texts)

        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

        return padded

    def fit_transform(self,texts:List[str]):
        """
        Fit tokenizer and transform texts
        """
        self.fit(texts)
        return self.transform(texts)

    def get_vocab_size(self) -> int:
        """
        Returns actual vocabulary size
        """
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)

if __name__ == "__main__":
    texts = [
        "apple releases new ai chip",
        "football team wins match",
        "stock market crashes today"
    ]

    tokenizer = LSTMTokenizer(vocab_size=1000, max_length=10)
    X = tokenizer.fit_transform(texts)

    print(X)
    print("Vocab size:", tokenizer.get_vocab_size())

        