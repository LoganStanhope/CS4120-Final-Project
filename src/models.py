from sklearn.linear_model import LogisticRegression
import torch


class GenericClfModel(object):

    def train(self):
        raise Exception("implement me in child classes")
    
    def predict(self):
        raise Exception("implement me in child classes")
    


class NaiveBayesClf(GenericClfModel):

    def __init__(self):
        GenericClfModel.__init__(self)

    def train(self):
        # training loop for naive bayes clf 
        pass

    def predict(self):
        # predictions for naive bayes
        pass 


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