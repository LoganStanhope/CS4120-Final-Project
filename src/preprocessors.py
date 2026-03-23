import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    def __init__(self, data):
        GenericPreprocessor.__init__(self)
        self.data = data

    def process_data(self):
        # implement BOW preprocessor
        # return processed data
        pass 


class TFIDFPreprocessor(GenericPreprocessor):
    def __init__(self, data: list[str]):
        GenericPreprocessor.__init__(self)
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english')


    def process_data(self):
        # implement TFIDF preprocessor
        # return processed data
        return self.vectorizer.fit_transform(self.data)

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