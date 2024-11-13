import os
import pandas as pd
import matplotlib.pyplot as plt
import time


def save_plots(filtered_data, filename):
    print("Started saving plots")
    # Create the folder to save plots if it doesn't exist
    folder_path = '/Users/ivanursul/Downloads/pi-plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create a figure with 4 subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: 'AccX', 'AccY', 'AccZ'
    axs[0, 0].plot(filtered_data['AccX'], label='AccX')
    axs[0, 0].plot(filtered_data['AccY'], label='AccY')
    axs[0, 0].plot(filtered_data['AccZ'], label='AccZ')
    axs[0, 0].set_title('Raw Accelerometer Data')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Acceleration (g)')
    axs[0, 0].legend()

    # Plot 2: 'acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered'
    axs[0, 1].plot(filtered_data['acc_x_filtered'], label='AccX Filtered')
    axs[0, 1].plot(filtered_data['acc_y_filtered'], label='AccY Filtered')
    axs[0, 1].plot(filtered_data['acc_z_filtered'], label='AccZ Filtered')
    axs[0, 1].set_title('Filtered Accelerometer Data')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Acceleration (g)')
    axs[0, 1].legend()

    # Plot 3: 'Magnitude', 'acc_magnitude_filtered'
    axs[1, 0].plot(filtered_data['Magnitude'], label='Raw Magnitude')
    axs[1, 0].plot(filtered_data['acc_magnitude_filtered'], label='Filtered Magnitude')
    axs[1, 0].set_title('Magnitude Comparison (Raw vs Filtered)')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Magnitude (g)')
    axs[1, 0].legend()

    # Plot 4: 'Altitude' and 'MovingAvgAltitude'
    axs[1, 1].plot(filtered_data['Altitude'], label='Altitude')
    axs[1, 1].plot(filtered_data['altitude_filtered'], label='Altitude filtered', color='orange', linewidth=2)
    axs[1, 1].set_title('Altitude Data and Moving Average')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Altitude (m)')
    axs[1, 1].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the figure as a single PNG file
    plt.savefig(os.path.join(folder_path, f'{filename}.png'))
    plt.close()

def read_csv_files_and_plot(folder_path):
    # Iterate over all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            try:
                data = pd.read_csv(file_path)
                print(f"Processing {filename}")

                # Ensure the necessary columns exist
                required_columns = ['AccX', 'AccY', 'AccZ', 'acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered',
                                    'Magnitude', 'acc_magnitude_filtered', 'Altitude']

                if all(col in data.columns for col in required_columns):
                    # Pass the filtered data to the save_plots function
                    save_plots(data, os.path.splitext(filename)[0])
                else:
                    print(f"Skipping {filename}: Missing required columns")

                time.sleep(1)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Call the method to read files and generate plots
folder_path = '/Users/ivanursul/Downloads/accelerometer_data_raw'  # Replace with your folder path
read_csv_files_and_plot(folder_path)
