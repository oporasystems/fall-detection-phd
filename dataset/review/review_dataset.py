import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

class CSVViewer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.i = 0  # Current index in the CSV files list

    def next_file(self, event):
        self.i += 1
        plt.close()

    def previous_file(self, event):
        # Go to the previous file if possible
        if self.i > 0:
            self.i -= 1
        plt.close()

    def delete_file(self, event):
        file_to_delete = os.path.join(self.folder_path, self.csv_files[self.i])
        os.remove(file_to_delete)
        print(f"Deleted {self.csv_files[self.i]}")
        del self.csv_files[self.i]
        # Adjust index if it goes out of bounds after deletion
        if self.i >= len(self.csv_files):
            self.i = len(self.csv_files) - 1
        plt.close()

    def show_plots(self):
        while self.i < len(self.csv_files):
            filename = self.csv_files[self.i]
            file_path = os.path.join(self.folder_path, filename)
            print(f"Processing {filename}")
            try:
                data = pd.read_csv(file_path)

                # Ensure the necessary columns exist
                required_columns = [
                    'AccX', 'AccY', 'AccZ',
                    'acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered',
                    'Magnitude', 'acc_magnitude_filtered',
                    'Altitude', 'altitude_filtered'
                ]

                if all(col in data.columns for col in required_columns):
                    # Create a figure with 4 subplots (2x2 grid)
                    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

                    # Get screen dimensions and adjust figure size
                    screen_width, screen_height = plt.get_current_fig_manager().canvas.get_width_height()
                    fig.set_size_inches(screen_width / fig.dpi * 0.9, screen_height / fig.dpi * 0.9)  # Stretch to 90%

                    # Plot 1: 'AccX', 'AccY', 'AccZ'
                    axs[0, 0].plot(data['AccX'], label='AccX')
                    axs[0, 0].plot(data['AccY'], label='AccY')
                    axs[0, 0].plot(data['AccZ'], label='AccZ')
                    axs[0, 0].set_title('Raw Accelerometer Data')
                    axs[0, 0].set_xlabel('Time')
                    axs[0, 0].set_ylabel('Acceleration (g)')
                    axs[0, 0].legend()

                    # Plot 2: 'acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered'
                    axs[0, 1].plot(data['acc_x_filtered'], label='AccX Filtered')
                    axs[0, 1].plot(data['acc_y_filtered'], label='AccY Filtered')
                    axs[0, 1].plot(data['acc_z_filtered'], label='AccZ Filtered')
                    axs[0, 1].set_title('Filtered Accelerometer Data')
                    axs[0, 1].set_xlabel('Time')
                    axs[0, 1].set_ylabel('Acceleration (g)')
                    axs[0, 1].legend()

                    # Plot 3: 'Magnitude', 'acc_magnitude_filtered'
                    axs[1, 0].plot(data['Magnitude'], label='Raw Magnitude')
                    axs[1, 0].plot(data['acc_magnitude_filtered'], label='Filtered Magnitude')
                    axs[1, 0].set_title('Magnitude Comparison (Raw vs Filtered)')
                    axs[1, 0].set_xlabel('Time')
                    axs[1, 0].set_ylabel('Magnitude (g)')
                    axs[1, 0].legend()

                    # Plot 4: 'Altitude' and 'altitude_filtered'
                    axs[1, 1].plot(data['Altitude'], label='Altitude')
                    axs[1, 1].plot(data['altitude_filtered'], label='Altitude Filtered', color='orange', linewidth=2)
                    axs[1, 1].set_title('Altitude Data and Moving Average')
                    axs[1, 1].set_xlabel('Time')
                    axs[1, 1].set_ylabel('Altitude (m)')
                    axs[1, 1].legend()

                    # Adjust layout to prevent overlapping
                    plt.tight_layout()

                    # Create buttons
                    ax_prev = plt.axes([0.48, 0.01, 0.1, 0.05])
                    ax_next = plt.axes([0.59, 0.01, 0.1, 0.05])
                    ax_delete = plt.axes([0.7, 0.01, 0.1, 0.05])

                    btn_previous = Button(ax_prev, 'Previous')
                    btn_next = Button(ax_next, 'Next')
                    btn_delete = Button(ax_delete, 'Delete')

                    # Assign callbacks
                    btn_previous.on_clicked(self.previous_file)
                    btn_next.on_clicked(self.next_file)
                    btn_delete.on_clicked(self.delete_file)

                    # Display the plot and wait for user interaction
                    plt.show()
                else:
                    print(f"Skipping {filename}: Missing required columns")
                    self.i += 1  # Move to the next file

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                self.i += 1  # Move to the next file


if __name__ == "__main__":
    folder_path = '/Users/ivanursul/Documents/Dataset V4/Unprocessed_Falls'  # Replace with your folder path
    viewer = CSVViewer(folder_path)
    viewer.show_plots()
