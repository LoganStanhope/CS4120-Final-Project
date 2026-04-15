from src.models import *
from src.preprocessors import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_all_data

def run_pipeline(X, y, preprocessor_cls, pp_kwargs, pp_name, model_cls, model_kwargs, model_name):
    """
    Runs a full training and testing pipeline for a given preprocessor/model combination.
    Splits raw text first, then fits the preprocessor on train only to avoid data leakage.
    Both the preprocessor and model are instantiated fresh each run.
    """
    # 1. Split raw text first
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # 2. Fresh preprocessor fitted only on train data
    preprocessor = preprocessor_cls(data=X_train_raw, **pp_kwargs)
    X_train = preprocessor.process_data()        # fit_transform on train
    X_test  = preprocessor.transform(X_test_raw) # transform-only on test

    vocab_size = preprocessor.get_vocab_size()

    # 3. Inject vocab/input size then instantiate a fresh model
    if model_name == 'mlp':
        model = model_cls(input_dim=vocab_size, **model_kwargs)
    elif model_name == 'rnn':
        model = model_cls(vocab_size=vocab_size, **model_kwargs)
    else:
        model = model_cls(**model_kwargs)

    # 4. Train and evaluate
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


def main():
    # Load and label data
    # true_df = pd.read_csv("data/True.csv")
    # true_df['label'] = 0  # 0 = true
    # fake_df = pd.read_csv("data/Fake.csv")
    # fake_df['label'] = 1  # 1 = fake

    # full_df = pd.concat([true_df, fake_df])
    # full_df.drop(columns=['date'], inplace=True)
    full_df = load_all_data()

    print(full_df.shape)
    print(full_df['label'].value_counts())

    y = full_df['label']
    X = full_df.drop(columns=['label'])

    # Series[str] for preprocessors that work on a single text column
    X_text = X.apply(lambda row: " ".join(row.astype(str)), axis=1)

    # (class, kwargs) — no data attached, no instances created yet
    preprocessor_config = {
        'simple': (SimplePreprocessor, {}),
        'bow':    (BOWPreprocessor,    {'max_features': 20_000}),
        'tfidf':  (TFIDFPreprocessor,  {'columns': X.columns}),
        'ngram':  (NGramPreprocessor,  {'n': 2, 'max_features': 10_000}),
    }

    model_config = {
        'logreg':      (LogRegClf,        {}),
        'naive_bayes': (NaiveBayesClf,    {}),
        'mlp':         (MLPClf,           {
            # input_dim injected at runtime from vocab_size
            'output_dim':  2,
            'hidden_dims': [256, 128],
            'dropout':     0.3,
            'lr':          1e-3,
            'epochs':      10,
            'batch_size':  32,
        }),
        'rnn':         (NeuralNetworkClf, {
            # vocab_size injected at runtime from preprocessor
            'embed_dim':   64,
            'hidden_dim':  64,
            'output_dim':  2,
            'num_layers':  1,
            'dropout':     0.3,
            'lr':          1e-3,
            'epochs':      5,
            'batch_size':  128,
            'max_seq_len': 128,
        }),
    }

    for pp_name, (preprocessor_cls, pp_kwargs) in preprocessor_config.items():
        for model_name, (model_cls, model_kwargs) in model_config.items():
            print(f"======= {pp_name} + {model_name} =======")

            # TFIDFPreprocessor needs the full DataFrame; others use X_text
            X_input = X if pp_name == 'tfidf' else X_text

            run_pipeline(
                X_input, y,
                preprocessor_cls, pp_kwargs, pp_name,
                model_cls, model_kwargs, model_name,
            )


if __name__ == '__main__':
    main()