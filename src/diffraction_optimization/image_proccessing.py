import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def parse_mnist_digit_to_matrix(dataset_digit_vector: pd.Series) -> np.ndarray:
    # Remove first value (which is the label) and reshape to ndarray of 28x28
    return dataset_digit_vector[1:].to_numpy().reshape(28, 28)


def display_digit(digit_matrix: np.ndarray) -> None:
    plt.imshow(digit_matrix)
    plt.show()
