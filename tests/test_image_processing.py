from random import randint

import numpy as np
import pandas as pd
from pytest import fixture

from diffraction_optimization.file_io import (
    get_vectors_of_specific_digit,
    load_dataset_csv_to_df,
)
from diffraction_optimization.image_proccessing import (
    generate_random_phase_mask,
    grayscale_to_phase,
    parse_mnist_digit_to_matrix,
)


@fixture
def load_dataset() -> pd.DataFrame:
    return load_dataset_csv_to_df("src/diffraction_optimization/assets/mnist_test.csv")


@fixture
def get_digit(load_dataset) -> np.ndarray:
    dataset = load_dataset
    digit_to_select = randint(0, 9)
    digits = get_vectors_of_specific_digit(dataset, digit_to_select)
    return parse_mnist_digit_to_matrix(digits.iloc[0]), digit_to_select


def test_parse_mnist_digit_to_matrix(get_digit):
    digit, _ = get_digit
    assert digit.shape == (28, 28)


def test_grayscale_to_phase(get_digit):
    digit, _ = get_digit
    phase_digit = grayscale_to_phase(digit)
    assert np.max(phase_digit) <= 2 * np.pi
    assert np.min(phase_digit) >= 0


def test_generate_random_mask():
    random_mask = generate_random_phase_mask()
    assert random_mask.shape == (28, 28)
