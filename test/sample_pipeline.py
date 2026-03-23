import sys, os
# Navigate one level up from test/ to the project root (CS4120-Final-Project/)
# and insert it at the front of Python's module search path.
# This ensures `from src.models import *` resolves correctly regardless of
# where the script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import *
from src.preprocessors import * 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create dataset 
true_df = pd.read_csv("data/True.csv")
true_df['label'] = 0
fake_df = pd.read_csv("data/Fake.csv")
fake_df['label'] = 1

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

# apply tfidf vectorizer 
## tfidf can only handle one column at a time, but we have multiple 
## text columns

preprocessor = TFIDFPreprocessor(data=X, columns=X.columns)
X_processed = preprocessor.process_data()
print(X_processed.shape)


# train on NB classifier

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
clf = NaiveBayesClf()
clf.train(X_train, y_train)
preds = clf.predict(X_test)

accuracy = accuracy_score(y_test, preds)
# 0.9994060876020787
print(accuracy)

# ============================================================
# BOW + MLP
# ============================================================

# BOW only handles a single text column, so concatenate all text columns into one
X_text = X.apply(lambda row: " ".join(row.astype(str)), axis=1)
bow_preprocessor = BOWPreprocessor(data=X_text)

X_bow = bow_preprocessor.process_data()
X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.3, random_state=42
)

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
print(accuracy)