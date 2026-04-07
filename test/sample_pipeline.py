import sys, os
# Navigate one level up from test/ to the project root (CS4120-Final-Project/)
# and insert it at the front of Python's module search path.
# This ensures `from src.models import *` resolves correctly regardless of
# where the script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import *
from src.preprocessors import * 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Load & prepare data
# ============================================================
true_df = pd.read_csv("data/True.csv")
true_df['label'] = 0 # 0 is true
fake_df = pd.read_csv("data/Fake.csv")
fake_df['label'] = 1 # 1 is fake

## 1 = fake, 0 = true

## combine dfs together and shuffle
full_df = pd.concat([true_df, fake_df])
print(full_df['label'].value_counts())
##############
# 1    23481
# 0    21417
############## - balanced dataset

print(full_df.shape)
print(full_df.columns)


# keep relevant features 
full_df.drop(columns=['date'], inplace=True)

# isolate data from label
y = full_df['label']
X = full_df.drop(columns=['label'])

# Concatenate all text columns into a single string per article.
# Both NGramPreprocessor and the RNN vocab builder work on a Series[str].
X_text = X.apply(lambda row: " ".join(row.astype(str)), axis=1)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# ============================================================
# TF-IDF + NB
# ============================================================
preprocessor = TFIDFPreprocessor(data=X, columns=X.columns)
X_processed = preprocessor.process_data()
X_train_tfidf, X_test_tfidf, y_train_nb, y_test_nb = train_test_split(
    X_processed, y, test_size=0.3, random_state=42
)
clf = NaiveBayesClf()
clf.train(X_train_tfidf, y_train_nb)
preds = clf.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test_nb, preds)
# 0.9994060876020787
print(f"TF-IDF + NaiveBayes accuracy: {nb_accuracy:.8f}")

# ============================================================
# BOW + MLP
# ============================================================

# BOW only handles a single text column, so concatenate all text columns into one
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
# 0.99606533
print(f"BOW + MLP accuracy: {mlp_accuracy:.8f}")

# ============================================================
# N-Gram(?) + RNN
# ============================================================

# RNN pretty much just ignores the NN preprocessor since it just uses it to
# be a vocab source which any preprocessor can be used for.
ngram_pre = NGramPreprocessor(data=X_train_text, n=1, max_features=10_000)
ngram_pre.process_data()
vocab = ngram_pre.get_vocab()

MAX_SEQ_LEN = 128  # keep this small so we can actually run the code X_X
X_train_seq = ngram_pre.to_sequences(X_train_text.tolist(), max_seq_len=MAX_SEQ_LEN)
X_test_seq  = ngram_pre.to_sequences(X_test_text.tolist(),  max_seq_len=MAX_SEQ_LEN)

nn_clf = NeuralNetworkClf(
    vocab_size=len(vocab),
    embed_dim=64,       # small
    hidden_dim=64,      # small
    output_dim=2,
    num_layers=1,       # single layer — much faster than 2
    dropout=0.3,
    lr=1e-3,
    epochs=5,
    batch_size=128,     # larger batch = fewer steps per epoch
    max_seq_len=MAX_SEQ_LEN,
)
nn_clf.train(X_train_seq, y_train.to_numpy())
nn_preds = nn_clf.predict(X_test_seq)
nn_accuracy = accuracy_score(y_test, nn_preds)
# accuracy = 0.9972
print(f"\nN-Gram + RNN accuracy: {nn_accuracy:.8f}")
# to see precision, f1, recall, etc.
print(classification_report(y_test, nn_preds, target_names=["real", "fake"]))

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
#accuracy = 0.99231626
print(f"\nsimple + logistic regression accuracy: {logreg_accuracy:.8f}")