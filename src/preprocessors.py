import nltk


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