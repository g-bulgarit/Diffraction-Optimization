import numpy as np
from matplotlib import pyplot as plt
from diffraction_optimization.diffraction_model import DiffractionSystem
from diffraction_optimization.image_proccessing import (
    # generate_horizontal_line,
    # generate_random_phase_mask,
    mutate_phase_mask,
)
from diffraction_optimization.file_io import (
    load_dataset_csv_to_df,
    load_digit_images_from_dataset,
)

mnist_dataset = load_dataset_csv_to_df(
    "src/diffraction_optimization/assets/mnist_test.csv"
)
zero_digits = load_digit_images_from_dataset(mnist_dataset, 0, load_amt=50)
one_digits = load_digit_images_from_dataset(mnist_dataset, 1, load_amt=50)
zero_predicition_rates = list()
one_predicition_rates = list()
average_prediction_rates = list()


def accept_bad_soltuion(temperature: float) -> bool:
    if temperature * np.random.rand() > 0.4:
        return True
    return False


def simulated_annealing_optimization():
    # Define system hyperparameters
    output_plane_distance_mm = 1000

    # Define algo hyperparameters
    temperature = 500
    max_num_iterations = 100
    bad_solutions_before_restart = 10

    # Initialize simulation parameters
    num_iterations = 0
    bad_solution_counter = 0
    # phase_mask = generate_horizontal_line()
    phase_mask = np.zeros((28, 28))
    best_phase_mask = phase_mask

    system = DiffractionSystem(
        screen_height_mm=output_plane_distance_mm // 100,
        screen_width_mm=output_plane_distance_mm // 100,
        num_samples_x=121,
        num_samples_y=121,
        screen_position_z_mm=output_plane_distance_mm,
    )

    # Initialize system
    system.set_phase_mask(phase_mask)
    zero_prediction_rate, one_prediction_rate = system.generate_predictions(
        zero_images=zero_digits, one_images=one_digits
    )
    zero_predicition_rates.append(zero_prediction_rate)
    one_predicition_rates.append(one_prediction_rate)
    avg_prediction_rate = (zero_prediction_rate + one_prediction_rate) / 2
    average_prediction_rates.append(avg_prediction_rate)

    while temperature > 0 and num_iterations < max_num_iterations:
        # Check if we need to reset:
        if bad_solution_counter >= bad_solutions_before_restart:
            bad_solution_counter = 0
            phase_mask = best_phase_mask
        else:
            # Mutate phase mask
            new_phase_mask = mutate_phase_mask(phase_mask)
        system.set_phase_mask(new_phase_mask)
        zero_prediction_rate, one_prediction_rate = system.generate_predictions(
            zero_images=zero_digits, one_images=one_digits
        )
        zero_predicition_rates.append(zero_prediction_rate)
        one_predicition_rates.append(one_prediction_rate)
        avg_prediction_rate = (zero_prediction_rate + one_prediction_rate) / 2
        average_prediction_rates.append(avg_prediction_rate)

        # Calculate new temperature
        temperature = 1 - ((num_iterations + 1) / max_num_iterations)

        if avg_prediction_rate <= average_prediction_rates[-2]:
            # If the current prediction is worse than the previous - consider the temperature
            bad_solution_counter += 1
            if accept_bad_soltuion(temperature):
                # If the temperature allows - accept the bad soltuion even if it's bad for us
                phase_mask = new_phase_mask
        else:
            phase_mask = new_phase_mask
            if avg_prediction_rate >= max(average_prediction_rates):
                best_phase_mask = phase_mask

        num_iterations += 1

    _, axes = plt.subplots(2, 1)
    axes[0].plot(zero_predicition_rates, label="Zero: True Positive Rate")
    axes[0].plot(one_predicition_rates, label="One: True Positive Rate")
    axes[0].plot(average_prediction_rates, label="Average True Positive Rate")
    axes[0].legend()
    axes[0].set_xlabel("Iteration [#]")
    axes[0].set_ylabel("Correct Prediction [%]")

    axes[1].imshow(phase_mask)
    axes[1].set_title("Final Phase Mask")
    plt.show()
    pass

    # Now evaluate on new data:
    zero_digits_test_set = load_digit_images_from_dataset(
        mnist_dataset, 0, load_amt=200
    )[:, :, 51:]
    one_digits_test_set = load_digit_images_from_dataset(
        mnist_dataset, 1, load_amt=200
    )[:, :, 51:]

    (
        zero_true_positive_rate,
        one_true_positive_rate,
    ) = system.generate_predictions(
        zero_images=zero_digits_test_set, one_images=one_digits_test_set
    )

    plt.figure()
    plt.title("Classification Results on Test Set")
    plt.bar("Zero True Positive Rate", zero_true_positive_rate)
    plt.bar("One True Positive Rate", one_true_positive_rate)
    plt.show()
    pass


if __name__ == "__main__":
    simulated_annealing_optimization()
