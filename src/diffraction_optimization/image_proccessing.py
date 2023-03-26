import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_mnist_digit_to_matrix(dataset_digit_vector: pd.Series) -> np.ndarray:
    # Remove first value (which is the label) and reshape to ndarray of 28x28
    return dataset_digit_vector[1:].to_numpy().reshape(28, 28)


def grayscale_to_phase(input_img: np.ndarray) -> np.ndarray:
    return (input_img / 255) * 2 * np.pi


def generate_random_phase_mask():
    return np.random.rand(28, 28) * 2 * np.pi


def generate_slit_phase_mask(width: int, height: int, steps: int):
    phase_mask = np.zeros((28, 28))
    for idx in range(0, 28, steps):
        h = (28 - height) // 2
        phase_mask[h : 28 - h, idx : idx + width] = 2 * np.pi
    return phase_mask


def generate_donut_phase_mask(inner_radius: int, outer_radius: int) -> np.ndarray:
    phase_mask = np.zeros((28, 28))
    for ix in range(28):
        for iy in range(28):
            point_radius = np.sqrt((ix - 14) ** 2 + (iy - 14) ** 2)
            if point_radius < outer_radius and point_radius > inner_radius:
                phase_mask[ix, iy] = 2 * np.pi
    return phase_mask


def generate_horizontal_line() -> np.ndarray:
    phase_mask = np.zeros((28, 28))
    phase_mask[13:15, :] = 1
    return phase_mask


def display_digit(digit_matrix: np.ndarray, digit_label=None) -> None:
    plt.imshow(digit_matrix)
    if digit_label:
        plt.title(f"{digit_label}")
    plt.colorbar()
    plt.show()
