import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


class GenericPreprocessor(object):
  
    def process_data(self):
        raise Exception("implement me in my subclasses!")

        

class SimplePreprocessor(GenericPreprocessor):
    def __init__(self, data):
        GenericPreprocessor.__init__(self)
        self.data = data

    def process_data(self):
        # implement simple preprocessor
        # return processed data 

        pass 


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
            strip_accents="unicode", # - strip_accents: normalizes unicode characters (e.g. é -> e)
            lowercase=True, # - lowercase: ensures case-insensitive vocabulary
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", # - token_pattern: only keeps alphabetic tokens of 2+ characters, dropping numbers/punctuation
        )

    """
    Fit the CountVectorizer to the data and transform it into a BoW representation.
        Returns:
            A numpy array of shape (n_samples, vocab_size) containing the BoW representation of the input data.
    """
    def process_data(self):
        return self.vectorizer.fit_transform(self.data).toarray().astype(np.float32)

    def transform(self, texts):
        """Transform unseen texts using the fitted vocabulary."""
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    """Get the size of the vocabulary learned by the CountVectorizer."""
    def vocab_size(self):
        """
        Number of tokens in the fitted vocabulary.
        Useful for defining input dimensions of downstream models.
        """
        return len(self.vectorizer.vocabulary_)


class TFIDFPreprocessor(GenericPreprocessor):
    def __init__(self, data, columns):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.columns = columns

    def process_data(self):
        # implement TFIDF preprocessor
        # return processed data
        transformer = ColumnTransformer(
            transformers = [
                (f"{col}_tfidf", TfidfVectorizer(stop_words='english'), col)
                for col in self.columns
            ], 
            remainder='passthrough')
        return transformer.fit_transform(self.data)

class NGramPreprocessor(GenericPreprocessor):
    def __init__(self, data, n):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.n = n

    def process_data(self):
        # implement n-gram preprocessor
        # return processed data
        pass 


def main():
    # sanity check for each preprocessor
    true_df = pd.read_csv("../data/True.csv")
    lines = list(true_df['title'])

    preprocessor = TFIDFPreprocessor(lines)
    print(len(preprocessor.data))

    # process data
    processed_data = preprocessor.process_data()
    print(processed_data.shape)
    print(processed_data)


if __name__ == "__main__":
    main()