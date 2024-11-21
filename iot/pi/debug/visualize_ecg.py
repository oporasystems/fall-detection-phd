import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# Directory containing the CSV files
CSV_DIRECTORY = "/Users/ivanursul/Downloads/ecg_recordings"

# Sampling rate (Hz) used during recording
SAMPLE_RATE = 100  # Adjust if different

# Function to process and visualize ECG data from a CSV file
def visualize_ecg(csv_filepath):
    # Load the data
    data = pd.read_csv(csv_filepath)

    # Extract timestamp and voltage
    timestamps = data['Timestamp'].values
    voltages = data['Voltage'].values

    # Remove DC offset
    voltages = voltages - np.mean(voltages)

    # Apply bandpass filter to remove noise (e.g., 0.5 - 40 Hz)
    fs = SAMPLE_RATE  # Sampling frequency
    lowcut = 0.5
    highcut = 40.0

    # Design the filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(2, [low, high], btype='band')

    # Apply the filter
    filtered_voltages = signal.filtfilt(b, a, voltages)

    # Detect R-peaks (QRS complexes)
    peak_height = 0.5 * np.std(filtered_voltages)
    min_distance = int(fs * 0.6)  # Minimum samples between peaks (assuming minimum heart rate of 100 bpm)
    peaks, _ = signal.find_peaks(
        filtered_voltages,
        distance=min_distance,
        height=peak_height
    )

    # Plot the ECG signal
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, filtered_voltages, label='Filtered ECG Signal', color='blue')

    # Mark detected peaks
    plt.plot(timestamps[peaks], filtered_voltages[peaks], 'ro', label='Detected R-peaks')

    # Add labels and title
    plt.title(f'ECG Signal from {os.path.basename(csv_filepath)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally, print estimated heart rate
    if len(peaks) > 1:
        rr_intervals = np.diff(timestamps[peaks])  # Time between successive peaks
        avg_rr_interval = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr_interval
        print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
    else:
        print("Not enough peaks detected to estimate heart rate.")

# List all CSV files in the directory
csv_files = [f for f in os.listdir(CSV_DIRECTORY) if f.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    csv_filepath = os.path.join(CSV_DIRECTORY, csv_file)
    print(f"\nProcessing {csv_file}...")
    visualize_ecg(csv_filepath)
