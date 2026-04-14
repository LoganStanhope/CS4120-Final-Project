import sys, os
# Navigate one level up from test/ to the project root
# so `from src.models import *` resolves correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import *
from src.preprocessors import *
from src.data_loader import load_all_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Load & prepare data
# ============================================================
full_df = load_all_data(
    kaggle_true="data/True.csv",          # set to None to skip Kaggle
    kaggle_fake="data/Fake.csv",          # set to None to skip Kaggle
    liar_dir="data/liar_dataset",         # set to None to skip LIAR
    include_ambiguous=False,              # drop half-true / barely-true
)

## 1 = fake, 0 = real
print(full_df['label'].value_counts())

##############
# Combined dataset no ambiguous
# 1    27035
# 0    25924
# Kaggle dataset
# 1    23481
# 0    21417
# Liar dataset no ambiguous
# 1    4507
# 0    3554
############## - balanced dataset

print(full_df.shape)
print(full_df.columns)

# ── Isolate text and labels ──
y = full_df['label']
X_text = full_df['text']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# ============================================================
# TF-IDF + NB
# ============================================================
X_tfidf_df = full_df[['text']]
preprocessor = TFIDFPreprocessor(data=X_tfidf_df, columns=['text'])
X_processed = preprocessor.process_data()
X_train_tfidf, X_test_tfidf, y_train_nb, y_test_nb = train_test_split(
    X_processed, y, test_size=0.3, random_state=42
)
clf = NaiveBayesClf()
clf.train(X_train_tfidf, y_train_nb)
preds = clf.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test_nb, preds)
print(f"TF-IDF + NaiveBayes accuracy: {nb_accuracy:.8f}")

# ============================================================
# BOW + MLP
# ============================================================
bow_preprocessor = BOWPreprocessor(data=X_train_text)
X_train_bow = bow_preprocessor.process_data()
X_test_bow = bow_preprocessor.transform(X_test_text)

input_dim = bow_preprocessor.vocab_size()   # vocab size from fitted BOW
output_dim = 2                              # binary classification: fake vs true

mlp_clf = MLPClf(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dims=[256, 128],
    dropout=0.3,
    lr=1e-3,
    epochs=10,
    batch_size=32
)

mlp_clf.train(X_train_bow, y_train.to_numpy())
mlp_preds = mlp_clf.predict(X_test_bow)
mlp_accuracy = accuracy_score(y_test, mlp_preds)
print(f"BOW + MLP accuracy: {mlp_accuracy:.8f}")

# ============================================================
# N-Gram + RNN
# ============================================================

# RNN pretty much just ignores the NN preprocessor since it just uses it to
# be a vocab source which any preprocessor can be used for.
ngram_pre = NGramPreprocessor(data=X_train_text, n=1, max_features=10_000)
ngram_pre.process_data()
vocab = ngram_pre.get_vocab()

MAX_SEQ_LEN = 128  # keep this small so we can actually run the code
X_train_seq = ngram_pre.to_sequences(X_train_text.tolist(), max_seq_len=MAX_SEQ_LEN)
X_test_seq  = ngram_pre.to_sequences(X_test_text.tolist(),  max_seq_len=MAX_SEQ_LEN)

nn_clf = NeuralNetworkClf(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=64,
    output_dim=2,
    num_layers=1,
    dropout=0.3,
    lr=1e-3,
    epochs=5,
    batch_size=128,
    max_seq_len=MAX_SEQ_LEN,
)
nn_clf.train(X_train_seq, y_train.to_numpy())
nn_preds = nn_clf.predict(X_test_seq)
nn_accuracy = accuracy_score(y_test, nn_preds)
print(f"\nN-Gram + RNN accuracy: {nn_accuracy:.8f}")
print(classification_report(y_test, nn_preds, target_names=["real", "fake"])) # to see precision, f1, recall, etc.

# ============================================================
# Simple + Logistic Regression
# ============================================================
#preprocess text
simple_pre = SimplePreprocessor(data=X_train_text)
X_train_simple = simple_pre.process_data()
X_test_simple = SimplePreprocessor(data=X_test_text).process_data()
#converting to numeric features 
#use TF-IDF as a vectorizer only
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_simple)
X_test_vec = vectorizer.transform(X_test_simple)
#train model
logreg_clf = LogRegClf()
logreg_clf.train(X_train_vec, y_train)
#predicting + evaluate
logreg_preds = logreg_clf.predict(X_test_vec)
logreg_accuracy = accuracy_score(y_test, logreg_preds)
print(f"\nSimple + Logistic Regression accuracy: {logreg_accuracy:.8f}")
