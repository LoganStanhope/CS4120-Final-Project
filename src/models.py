from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class RNNModel(nn.Module):
    """Bidirectional vanilla RNN for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_idx) # Can replace this with our own word embedding implementation later if not too complex
        self.embed_drop = nn.Dropout(dropout)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # help prevent overfitting
            bidirectional=True,
            nonlinearity='tanh',
        )
        self.rnn_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # since bidirectional just concats the two hidden state sets and classifies

    def forward(self, x):
        # x = (batch, seq_len)
        embedded = self.embed_drop(self.embedding(x))

        # output = (batch, seq_len, hidden_dim * 2)
        # hidden = (num_layers * 2, batch, hidden_dim)
        output, hidden = self.rnn(embedded)

        # grab the final forward and backward hidden states from the last layer
        # hidden[-2] = forward, hidden[-1] = backward
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        return self.fc(self.rnn_drop(last_hidden))

class NeuralNetworkClf(GenericClfModel):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, output_dim=2,
                 num_layers=1, dropout=0.3, pad_idx=0,
                 lr=1e-3, epochs=5, batch_size=128, max_seq_len=128):
        GenericClfModel.__init__(self)
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RNNModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []

    def train(self, X_train, y_train):
        X_tensor = torch.tensor(X_train, dtype=torch.long)
        y_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}")

    def predict(self, X_test):
        X_tensor = torch.tensor(X_test, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy()

class MLP(nn.Module):
    """Simple feedforward network with configurable hidden layers."""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train):
        """
        Fit the MLP on training data.

        Args:
            X_train: np.ndarray of shape (n_samples, input_dim) — e.g. BOW or TFIDF matrix.
            y_train: np.ndarray of shape (n_samples,) — integer class labels.
        """
        
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
    