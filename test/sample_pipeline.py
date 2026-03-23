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