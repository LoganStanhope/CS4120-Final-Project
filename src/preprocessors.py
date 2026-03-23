import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
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
    @property
    def vocab_size(self):
        return len(self.vectorizer.vocabulary_)


class TFIDFPreprocessor(GenericPreprocessor):
    def __init__(self, data):
        GenericPreprocessor.__init__(self)
        self.data = data

    def process_data(self):
        # implement TFIDF preprocessor
        # return processed data
        pass 

class NGramPreprocessor(GenericPreprocessor):
    def __init__(self, data, n):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.n = n

    def process_data(self):
        # implement n-gram preprocessor
        # return processed data
        pass 