from pytest import fixture
import pandas as pd
from diffraction_optimization.image_proccessing import parse_mnist_digit_to_matrix
from diffraction_optimization.file_io import load_dataset_csv_to_df, get_vectors_of_specific_digit


@fixture
def load_all_zero_digits_from_dataset() -> pd.DataFrame:
    dataset = load_dataset_csv_to_df("src/diffraction_optimization/assets/mnist_test.csv")
    return get_vectors_of_specific_digit(dataset, 0)


def test_parse_mnist_digit_to_matrix(load_all_zero_digits_from_dataset):
    zero_images = load_all_zero_digits_from_dataset
    example_digit = parse_mnist_digit_to_matrix(zero_images.iloc[0])
    assert example_digit.shape == (28, 28)
