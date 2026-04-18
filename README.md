**CS4120 Final Project - Fake News Detection Pipeline**

A modular NLP pipeline for binary fake news classification. The system benchmarks multiple preprocessor and model combinations against a labeled dataset of real and fake news articles.
Labels: 0 = real news, 1 = fake news

**Overview** 

The pipeline evaluates all valid combinations of 4 preprocessors and 4 model. Preprocessors are always fit exclusively on training data to prevent data leakage.

**Preprocessors**

Key → Class → Output → Description 

simple → Simple Preprocessor →  np.ndarray of cleaned strings → Lowercasing, punctuation removal, short/non-alpha word filtering 

bow → BOWPreprocessor → Dense float32 array → Bag-of-Words via CountVectorizer 

tfidf → TFIDFPreprocessor → Sparse matrix → Per-column TF-IDF via ColumnTransformer

ngram → NGramPreprocessor → Dense float32 array or integer sequences → TF-IDF with unigrams+bigrams and also exposes to_sequences() for the RNN

**Models**

Key → Class → Backend → Notes 

logreg → LogRegClf → scikit-learn → Logistic Regression

naive_bayes → NaiveBayesClf → scikit-learn → Multinomial Naive Bayes

mlp → MLPClf → PyTorch → Configurable feedforward network 

rnn → NeuralNetworkClf → PyTorch → Bidirectional RNN with learned embeddings

**Requirements**

Python 3.8+ recommended. Install dependencies:

	pip install numpy scipy scikit-learn torch pandas nltk

**Running the Sample Pipeline**

A self-contained example script is available in test/sample_pipeline.py. It demonstrates each of the four main preprocessor and model combinations individually with detailed comments:

	python test/sample_pipeline.py

Has to be run from the project root so that data/True.csv and data/Fake.csv resolve correctly.


