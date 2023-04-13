import pandas as pd
import numpy as np
from typing import List
from diffraction_optimization.image_proccessing import parse_mnist_digit_to_matrix


def load_dataset_csv_to_df(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def get_vectors_of_specific_digit(dataset: pd.DataFrame, digit: int) -> pd.DataFrame:
    return dataset[dataset["label"] == digit]


def load_digit_images_from_dataset(
    dataset: pd.DataFrame, digit: int, load_amt=None
) -> np.ndarray:
    digit_vectors = get_vectors_of_specific_digit(dataset, digit)
    if load_amt is None:
        vectors_amt = len(digit_vectors)
    else:
        vectors_amt = load_amt
    digit_matrix = np.zeros((28, 28, vectors_amt))
    for index in range(vectors_amt):
        digit_matrix[:, :, index] = parse_mnist_digit_to_matrix(
            digit_vectors.iloc[index]
        )
    return digit_matrix


def load_specific_digits(
    dataset: pd.DataFrame, digit_list: List[int], samples_per_digit: int
) -> np.ndarray:
    num_digits = len(digit_list)
    full_data = np.zeros((28, 28, samples_per_digit, num_digits))
    for digit_index in range(len(digit_list)):
        digit_to_load = digit_list[digit_index]
        full_data[:, :, :, digit_index] = load_digit_images_from_dataset(
            dataset, digit_to_load, samples_per_digit
        )
    return full_data


def save_final_phase_mask(phase_mask: np.ndarray, filepath: str) -> None:
    with open(filepath, "wb+") as wfp:
        np.save(wfp, phase_mask)
