import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_mnist_digit_to_matrix(dataset_digit_vector: pd.Series) -> np.ndarray:
    # Remove first value (which is the label) and reshape to ndarray of 28x28
    return dataset_digit_vector[1:].to_numpy().reshape(28, 28)


def grayscale_to_phase(input_img: np.ndarray) -> np.ndarray:
    return (input_img / 255) * 2 * np.pi


def display_digit(digit_matrix: np.ndarray, digit_label=None) -> None:
    plt.imshow(digit_matrix)
    if digit_label:
        plt.title(f"{digit_label}")
        plt.colorbar()
    plt.show()
