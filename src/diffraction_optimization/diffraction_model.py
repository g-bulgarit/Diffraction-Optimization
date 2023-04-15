import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, fftshift
from diffraction_optimization.file_io import (
    load_dataset_csv_to_df,
    get_vectors_of_specific_digit,
)
from diffraction_optimization.image_proccessing import (
    parse_mnist_digit_to_matrix,
    grayscale_to_phase,
    generate_horizontal_line,
    normalize_output_image,
)


class DiffractionSystem:
    def __init__(
        self,
        screen_width_mm: int,
        screen_height_mm: int,
        num_samples_x: int,
        num_samples_y: int,
        wavelength_mm: int = 0.0007,
        screen_position_z_mm: int = 1000,
    ) -> None:
        self.x_res = screen_width_mm / num_samples_x
        self.y_res = screen_height_mm / num_samples_y
        self.wavelength_mm = wavelength_mm
        self.x_axis = self.x_res * (np.arange(num_samples_x) - num_samples_x // 2)
        self.y_axis = self.y_res * (np.arange(num_samples_y) - num_samples_y // 2)
        self.xgrid, self.ygrid = np.meshgrid(self.x_axis, self.y_axis)
        self.screen_position_z_mm = screen_position_z_mm

        self.num_samples_x = int(num_samples_x)
        self.num_samples_y = int(num_samples_y)
        self.starting_plane = np.zeros((num_samples_x, num_samples_y))

        self.screen_resolution_x = (
            self.screen_position_z_mm * self.wavelength_mm / (2 * screen_width_mm)
        )
        self.screen_resolution_y = (
            self.screen_position_z_mm * self.wavelength_mm / (2 * screen_height_mm)
        )
        self.screen_grid_x = self.screen_resolution_x * (
            np.arange(self.num_samples_x) - self.num_samples_x // 2
        )
        self.screen_grid_y = self.screen_resolution_y * (
            np.arange(self.num_samples_y) - self.num_samples_y // 2
        )

    def set_input_image(self, input_image: np.ndarray) -> np.ndarray:
        self.image = self.center_image(input_image)

    def set_input_images(self, input_images: List[np.ndarray]) -> np.ndarray:
        input_images_to_save = []
        for image in input_images:
            input_images_to_save.append(self.center_image(image))
        self.input_images = input_images_to_save

    def set_phase_mask(self, phase_mask: np.ndarray) -> np.ndarray:
        self.phase_mask = self.center_image(phase_mask)

    def center_image(self, image) -> np.ndarray:
        centered_image = np.zeros(self.starting_plane.shape)
        center_mask = np.zeros(self.starting_plane.shape)
        ymask, xmask = center_mask.shape
        himg, wimg = image.shape
        x_left = xmask // 2 - wimg // 2
        y_top = ymask // 2 - himg // 2
        center_mask[y_top : y_top + himg, x_left : x_left + wimg] = 1
        np.place(centered_image, center_mask, image)
        return centered_image

    def calculate_image_at_output_plane(self, input_image: np.ndarray) -> np.ndarray:
        E = self.center_image(input_image) * np.exp(1j * self.phase_mask)
        k = 2 * np.pi / self.wavelength_mm
        propagated_fft = fft2(
            E
            * np.exp(
                1j
                * k
                / (2 * self.screen_position_z_mm)
                * (self.xgrid**2 + self.ygrid**2)
            )
        )
        return normalize_output_image(fftshift(propagated_fft))

    def predict(self, output_image: np.ndarray, method="count_light", **kwargs) -> int:
        digits: List[int] = kwargs["digits"]
        num_digits = len(digits)
        if method == "count_light":
            total_pixels_in_image = output_image.shape[0] * output_image.shape[1]
            full_white_pixel_value = 255 * total_pixels_in_image
            total_pixel_sum = np.sum(output_image)

            step = 255 // num_digits
            for i in range(1, num_digits):
                if (
                    total_pixel_sum
                    <= ((i * step) / 255) * full_white_pixel_value * 0.25
                ):
                    return digits[i - 1]
            return digits[-1]

        if method == "splitx":
            end_idx = output_image.shape[1]
            if (end_idx % 2 != 0):
                # The image size is odd
                end_idx = end_idx - 1

            image_to_calculate_on = output_image[:, :end_idx]
            step = image_to_calculate_on.shape[0] // num_digits
            # Assign each digit a slice of x values
            regional_values = list()
            for x_region in range(num_digits):
                value_in_region = np.sum(image_to_calculate_on[:, x_region : x_region + step])
                regional_values.append(value_in_region)
            max_idx = np.argmax(regional_values)
            return digits[max_idx]

    def generate_predictions(
        self,
        digits_matrix: np.ndarray,
        digits_list: List,
        prediction_method="count_light",
    ) -> Tuple[float, float]:
        num_digits = digits_matrix.shape[3]
        prediction_rate_vector = np.zeros((1, num_digits))

        for digit_index in range(num_digits):
            prediction_vector = list()
            for image_index in range(digits_matrix.shape[2]):
                image_to_predict = self.calculate_image_at_output_plane(
                    digits_matrix[:, :, image_index, digit_index]
                )
                res = self.predict(
                    image_to_predict, prediction_method, digits=digits_list
                )
                prediction_vector.append(res)
            correct_predictions = 0
            for value in prediction_vector:
                if value == digits_list[digit_index]:
                    correct_predictions += 1
            prediction_rate_vector[:, digit_index] = correct_predictions / len(
                prediction_vector
            )
        return prediction_rate_vector

    def plot_input_and_output(self) -> None:
        plot_limits = [
            self.screen_grid_x[0],
            self.screen_grid_x[-1] + self.screen_resolution_x,
            self.screen_grid_y[0],
            self.screen_grid_y[-1] + self.screen_resolution_y,
        ]
        _, axes = plt.subplots(3, 2)

        axes[0, 0].imshow(self.input_images[0], cmap="gray")
        axes[0, 0].set_title("Original Image, 0")

        axes[1, 0].imshow(self.phase_mask, cmap="gray")
        axes[1, 0].set_title("Phase Mask")

        axes[2, 0].imshow(
            np.abs(self.output_images[0]), extent=plot_limits, cmap="gray"
        )
        axes[2, 0].set_title("Output Image (absolute value), 0")

        axes[0, 1].imshow(self.input_images[1], cmap="gray")
        axes[0, 1].set_title("Original Image, 1")

        axes[1, 1].imshow(self.phase_mask, cmap="gray")
        axes[1, 1].set_title("Phase Mask")

        axes[2, 1].imshow(
            np.abs(self.output_images[1]), extent=plot_limits, cmap="gray"
        )
        axes[2, 1].set_title("Output Image (absolute value), 1")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = load_dataset_csv_to_df(
        "src/diffraction_optimization/assets/mnist_test.csv"
    )

    input_images = []
    for digit in [0, 1]:
        single_digit_vector = get_vectors_of_specific_digit(dataset, digit).iloc[888]
        img = parse_mnist_digit_to_matrix(single_digit_vector)
        phase_img = grayscale_to_phase(img)
        input_images.append(phase_img)

    # phase_mask = generate_random_phase_mask()
    # phase_mask = generate_donut_phase_mask(0, 4)
    phase_mask = generate_horizontal_line()
    # phase_mask = generate_slit_phase_mask(width=3, height=16, steps=7)
    # phase_mask = np.zeros((28, 28))
    # for idx in range(0, 27, 7):
    #     phase_mask[:, idx : idx + 3] = 2 * np.pi

    for distance_mm in [200, 1000]:
        system = DiffractionSystem(
            screen_height_mm=distance_mm // 100,
            screen_width_mm=distance_mm // 100,
            num_samples_x=121,
            num_samples_y=121,
            screen_position_z_mm=distance_mm,
        )
        # system.set_input_images(input_images)
        # system.set_phase_mask(phase_mask)
        # system.calculate_images_at_output_plane()
        # system.plot_input_and_output()
