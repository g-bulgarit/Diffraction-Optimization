import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List


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


def normalize_output_image(output_image: np.ndarray) -> np.ndarray:
    power_image = np.abs(output_image)
    return (255 * power_image / np.max(power_image)).astype(np.uint8)


def generate_horizontal_line() -> np.ndarray:
    phase_mask = np.zeros((28, 28))
    phase_mask[13:15, :] = np.pi
    return phase_mask


def mutate_phase_mask(phase_mask: np.ndarray, temperature: float) -> np.ndarray:
    size = int(temperature * 28)
    mutated_block = np.random.rand(size, size) * np.pi * 2 * temperature
    valid_start_points_w = []
    valid_start_points_h = []

    if phase_mask.shape[0] - size > 0:
        valid_start_points_w.extend(list(range(phase_mask.shape[0] - size)))
        valid_start_points_h.extend(list(range(phase_mask.shape[1] - size)))
    else:
        valid_start_points_w.append(0)
        valid_start_points_h.append(0)

    start_w = np.random.randint(len(valid_start_points_w))
    start_h = np.random.randint(len(valid_start_points_h))
    phase_mask[start_w : start_w + size, start_h : start_h + size] = mutated_block
    return phase_mask


def display_digit(digit_matrix: np.ndarray, digit_label=None) -> None:
    plt.imshow(digit_matrix)
    if digit_label:
        plt.title(f"{digit_label}")
    plt.colorbar()
    plt.show()


def draw_confusion_matrix(confusion_matrix: np.ndarray, digits: List):
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap=plt.cm.PiYG, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=confusion_matrix[i, j],
                va="center",
                ha="center",
                size="large",
            )
    ax.set_xticklabels([""]+digits)
    ax.set_yticklabels([""]+digits)
    plt.xlabel("Predictions", fontsize=14)
    plt.ylabel("Ground Truth", fontsize=14)
    plt.title("Digit Confusion Matrix", fontsize=14)
    plt.show()
