import re
import string
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextCleaner:
    """
    Text preprocessing for LSTM-based text classification
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        use_lemmatization: bool = True
    ):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words("english"))

        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers (optional but common for news)
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)

    def clean_corpus(self, texts: List[str]) -> List[str]:
        """
        Clean a list of text samples
        """
        return [self.clean_text(text) for text in texts]

if __name__ == "__main__":
    cleaner = TextCleaner()

    sample = "Apple's NEW AI chip!!! Beats competitors by 30% 🚀"
    print(cleaner.clean_text(sample))
