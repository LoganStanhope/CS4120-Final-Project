import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import re
import scipy.sparse as sp

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


class GenericPreprocessor(object):

    def process_data(self):
        raise Exception("implement me in my subclasses!")

    def transform(self, data):
        raise Exception("implement me in my subclasses!")

    def get_vocab_size(self):
        raise Exception("implement me in my subclasses!")


class SimplePreprocessor(GenericPreprocessor):
    def __init__(self, data, max_features=20_000, min_df=1):
        GenericPreprocessor.__init__(self)
        self.data = data
        self._vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            lowercase=True,
            strip_accents="unicode",
        )
        self._fitted = False

    def _clean_text(self, texts):
        processed = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = text.split()
            tokens = [word for word in tokens if len(word) > 2 and word.isalpha()]
            processed.append(" ".join(tokens))
        return processed

    def process_data(self):
        cleaned = self._clean_text(self.data)
        matrix = self._vectorizer.fit_transform(cleaned)
        self._fitted = True
        return matrix  

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Call process_data() before transform().")
        cleaned = self._clean_text(texts)
        return self._vectorizer.transform(cleaned)  

    def get_vocab_size(self):
        if not self._fitted:
            raise RuntimeError("Call process_data() before get_vocab_size().")
        return len(self._vectorizer.vocabulary_)


class BOWPreprocessor(GenericPreprocessor):
    """BoW preprocessor using sklearn's CountVectorizer."""
    def __init__(self, data, max_features=20_000, min_df=3, max_df=0.95, binary=False):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )

    def process_data(self):
        return self.vectorizer.fit_transform(self.data)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_vocab_size(self):
        return len(self.vectorizer.vocabulary_)


class TFIDFPreprocessor(GenericPreprocessor):
    def __init__(self, data, columns):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.columns = columns
        self._transformer = None

    def process_data(self):
        self._transformer = ColumnTransformer(
            transformers=[
                (f"{col}_tfidf", TfidfVectorizer(stop_words='english'), col)
                for col in self.columns
            ],
            remainder='drop',  
        )
        return self._transformer.fit_transform(self.data) 

    def get_vocab_size(self):
        if self._transformer is None:
            raise RuntimeError("Call process_data() before get_vocab_size().")
        total = 0
        for _, tfidf, _ in self._transformer.transformers_:
            if hasattr(tfidf, 'vocabulary_'):
                total += len(tfidf.vocabulary_)
        return total

    def transform(self, data):
        if self._transformer is None:
            raise RuntimeError("Call process_data() before transform().")
        return self._transformer.transform(data)  

class NGramPreprocessor(GenericPreprocessor):
    def __init__(self, data, n=2, max_features=20_000, min_df=3, max_df=0.95,
                 sublinear_tf=True):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.n = n

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, n),
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            stop_words="english",
        )

    def process_data(self):
        return self.vectorizer.fit_transform(self.data) 

    def transform(self, texts):
        return self.vectorizer.transform(texts) 

    def get_vocab_size(self):
        return len(self.vectorizer.vocabulary_)

    def get_vocab(self):
        vocab = {token: idx + 2 for token, idx in self.vectorizer.vocabulary_.items()}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        return vocab

    def to_sequences(self, texts, max_seq_len=512):
        """Convert texts to integer index sequences for RNN input."""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        vocab = self.get_vocab()
        UNK_IDX = vocab["<UNK>"]
        tokenize = self.vectorizer.build_analyzer()

        result = np.zeros((len(texts), max_seq_len), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = tokenize(str(text))[:max_seq_len]
            result[i, :len(tokens)] = [vocab.get(tok, UNK_IDX) for tok in tokens]
        return result

    def top_ngrams(self, n_top=20):
        feature_names = self.vectorizer.get_feature_names_out()
        X = self.vectorizer.transform(self.data)
        mean_scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[::-1][:n_top]
        return [(feature_names[i], mean_scores[i]) for i in top_indices]


def main():
    true_df = pd.read_csv("../data/True.csv")
    lines = list(true_df['title'])

    preprocessor = TFIDFPreprocessor(lines, columns=['title'])
    print(len(preprocessor.data))

    processed_data = preprocessor.process_data()
    print(processed_data.shape)
    print(processed_data)


if __name__ == "__main__":
    main()