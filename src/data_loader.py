import os
import pandas as pd

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context",
]

LIAR_BINARY_MAP = {
    "true": 0,           # real
    "mostly-true": 0,    # real
    "false": 1,          # fake
    "pants-fire": 1,     # fake
}

def load_liar(liar_dir, include_ambiguous=False):
    frames = []
    for filename in ("train.tsv", "valid.tsv", "test.tsv"):
        path = os.path.join(liar_dir, filename)
        df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
        frames.append(df)

    liar = pd.concat(frames, ignore_index=True)

    label_map = dict(LIAR_BINARY_MAP)
    if include_ambiguous:
        label_map["barely-true"] = 1
        label_map["half-true"] = 0

    liar = liar[liar["label"].isin(label_map)].copy()
    liar["label"] = liar["label"].map(label_map).astype(int)
    liar["text"] = liar["statement"].fillna("")

    return liar[["text", "label"]].reset_index(drop=True)

def load_kaggle(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    true_df["label"] = 0

    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 1

    kaggle = pd.concat([true_df, fake_df], ignore_index=True)

    kaggle["text"] = (
        kaggle["title"].fillna("") + " " + kaggle["text"].fillna("")
    ).str.strip()

    return kaggle[["text", "label"]].reset_index(drop=True)


def load_all_data(
    kaggle_true="data/True.csv",
    kaggle_fake="data/Fake.csv",
    liar_dir="data/liar_dataset",
    include_ambiguous=False,
    shuffle=True,
    random_state=42,
):
    dfs = []

    if kaggle_true and kaggle_fake:
        dfs.append(load_kaggle(kaggle_true, kaggle_fake))

    if liar_dir:
        dfs.append(load_liar(liar_dir, include_ambiguous=include_ambiguous))

    combined_df = pd.concat(dfs, ignore_index=True)

    if shuffle:
        combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return combined_df