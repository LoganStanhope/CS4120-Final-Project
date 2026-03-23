from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

class MLP(nn.Module):
    """Simple feedforward network with configurable hidden layers."""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(MLP, self).__init__()

        # Build layers dynamically from hidden_dims list
        # e.g. hidden_dims=[256, 128] creates two hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification layer — no activation, handled by loss function
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPClf(GenericClfModel):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.3,
                 lr=1e-3, epochs=10, batch_size=32):
        GenericClfModel.__init__(self)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(input_dim, hidden_dims, output_dim, dropout).to(self.device)

        # CrossEntropyLoss expects raw logits (no softmax in forward)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train):
        """
        Fit the MLP on training data.

        Args:
            X_train: np.ndarray of shape (n_samples, input_dim) — e.g. BOW or TFIDF matrix.
            y_train: np.ndarray of shape (n_samples,) — integer class labels.
        """
        # Convert numpy arrays to tensors and wrap in a DataLoader for batching
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} — Loss: {total_loss/len(loader):.4f}")

    def predict(self, X_test):
        """
        Predict class labels for unseen data.

        Args:
            X_test: np.ndarray of shape (n_samples, input_dim).

        Returns:
            np.ndarray of predicted integer class labels.
        """
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # argmax over logits gives the predicted class index
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy()
    