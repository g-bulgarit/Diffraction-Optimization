from pytest import fixture

from diffraction_optimization.file_io import (get_vectors_of_specific_digit,
                                              load_dataset_csv_to_df)


@fixture
def load_dataset():
    # Load the test dataset because it's more lean to reduce testing time.
    return load_dataset_csv_to_df("src/diffraction_optimization/assets/mnist_test.csv")


def test_filter_specific_digit(load_dataset):
    dataset = load_dataset
    for digit in range(9):
        images_df = get_vectors_of_specific_digit(dataset, digit)
        assert len(images_df) > 0
