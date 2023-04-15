import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from diffraction_optimization.diffraction_model import DiffractionSystem
from diffraction_optimization.image_proccessing import (
    mutate_phase_mask,
)
from diffraction_optimization.file_io import (
    load_dataset_csv_to_df,
    save_final_phase_mask,
    load_specific_digits,
)

mnist_dataset = load_dataset_csv_to_df(
    "src/diffraction_optimization/assets/mnist_test.csv"
)

digits_to_load = [0, 1]
dataset_matrix = load_specific_digits(mnist_dataset, digits_to_load, 200)

prediction_rates = list()
average_prediction_rates = list()


def accept_bad_soltuion(temperature: float) -> bool:
    if temperature * np.random.rand() > 0.4:
        return True
    return False


def simulated_annealing_optimization():
    # Define system hyperparameters
    output_plane_distance_mm = 1000

    # Define algo hyperparameters
    temperature = 1
    max_num_iterations = 300
    bad_solutions_before_restart = 10
    prediction_rates_over_time = np.zeros((max_num_iterations, len(digits_to_load)))

    # Initialize simulation parameters
    num_iterations = 0
    bad_solution_counter = 0
    # phase_mask = generate_horizontal_line()
    phase_mask = np.random.rand(28, 28)
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
    prediction_rates = system.generate_predictions(
        dataset_matrix, digits_to_load, prediction_method="count_light"
    )
    avg_prediction_rate = np.mean(prediction_rates)
    average_prediction_rates.append(avg_prediction_rate)

    while temperature > 0 and num_iterations < max_num_iterations:
        # Check if we need to reset:
        if bad_solution_counter >= bad_solutions_before_restart:
            bad_solution_counter = 0
            phase_mask = best_phase_mask
        else:
            # Mutate phase mask
            new_phase_mask = mutate_phase_mask(phase_mask, temperature)
        system.set_phase_mask(new_phase_mask)
        prediction_rates = system.generate_predictions(
            dataset_matrix, digits_to_load, prediction_method="count_light"
        )
        avg_prediction_rate = np.mean(prediction_rates)
        average_prediction_rates.append(avg_prediction_rate)
        prediction_rates_over_time[num_iterations, :] = prediction_rates

        if avg_prediction_rate > 0.97:
            # Training is good enough
            break

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

    final_predictions_over_time = prediction_rates_over_time[: num_iterations + 1, :]
    plt.plot(average_prediction_rates, label="Average Prediction Rate")
    for index, digit in enumerate(digits_to_load):
        plt.plot(
            final_predictions_over_time[:, index], label=f"{digit} Prediction Rate"
        )
    plt.title("Average correct prediction rates as a function of iteration number")
    plt.xlabel("Iteration [#]")
    plt.ylabel("Prediction rate")
    plt.legend()
    plt.show()

    timenow = datetime.strftime(datetime.now(), "%d_%m_%Y_%H_%M")
    output_path = (
        f"models/phase_mask_{timenow}_{int(round(avg_prediction_rate, 2) * 100)}.npy"
    )
    save_final_phase_mask(phase_mask, output_path)

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

    # Now evaluate on new data:
    evaluation_data_matrix = load_specific_digits(mnist_dataset, digits_to_load, 200)

    (
        zero_true_positive_rate,
        one_true_positive_rate,
    ) = system.generate_predictions(
        zero_images=evaluation_data_matrix[:, :, :, 0],
        one_images=evaluation_data_matrix[:, :, :, 1],
    )

    plt.figure()
    plt.title("Classification Results on Test Set")
    plt.bar("Zero True Positive Rate", zero_true_positive_rate)
    plt.bar("One True Positive Rate", one_true_positive_rate)
    plt.show()
    pass

    # Show results on zeros and ones
    comparisons_to_show = 5

    for _ in range(comparisons_to_show):
        random_index = np.random.randint(len(evaluation_data_matrix[:, :, :, 0]))
        zero_img = evaluation_data_matrix[:, :, random_index, 0]
        one_img = evaluation_data_matrix[:, :, random_index, 1]
        zero_output = system.calculate_image_at_output_plane(zero_img)
        one_output = system.calculate_image_at_output_plane(one_img)

        _, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(zero_img)
        axes[0, 0].set_title("Input Image: 0 Digit")
        axes[0, 1].imshow(zero_output)
        axes[0, 1].set_title("Output Image: 0 Digit")

        axes[1, 0].imshow(one_img)
        axes[1, 0].set_title("Input Image: 1 Digit")
        axes[1, 1].imshow(one_output)
        axes[1, 1].set_title("Output Image: 1 Digit")
        plt.show()


if __name__ == "__main__":
    simulated_annealing_optimization()
