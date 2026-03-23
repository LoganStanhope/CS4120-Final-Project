from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import torch


class GenericClfModel(object):

    def train(self):
        raise Exception("implement me in child classes")
    
    def predict(self):
        raise Exception("implement me in child classes")
    


class NaiveBayesClf(GenericClfModel):

    def __init__(self):
        GenericClfModel.__init__(self)
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        # training loop for naive bayes clf 
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # predictions for naive bayes
        return self.model.predict(X_test)


class LogRegClf(GenericClfModel):

    def __init__(self):
        GenericClfModel.__init__(self)
        # example...
        self.model = LogisticRegression()

    def train(self):
        # training for log reg 
        pass

    def predict(self):
        # prediction for log reg
        pass 


class NeuralNetworkClf(GenericClfModel):
    def __init__(self):
        GenericClfModel.__init__(self)

    def train(self):
        # training for NN
        pass

    def predict(self):
        # prediction for NN
        pass


class MLPClf(GenericClfModel):
    def __init__(self):
        GenericClfModel.__init__(self)

    def train(self):
        # training for MLP
        pass 

    def predict(self):
        # prediction for MLP
        pass