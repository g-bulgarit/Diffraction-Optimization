import pandas as pd


def load_dataset_csv_to_df(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def get_vectors_of_specific_digit(dataset: pd.DataFrame, digit: int) -> pd.DataFrame:
    return dataset[dataset['label'] == digit]
