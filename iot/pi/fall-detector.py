import time
import math
import torch
import torch.nn as nn
import psutil
import os
import pandas as pd
import smbus
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from performer_pytorch import Performer
import numpy as np
import json
from scipy.signal import butter, filtfilt
import RPi.GPIO as GPIO
from logging_config import setup_logging

# Set up logging
logger = setup_logging("fall-detector")

# Create a new Mpu6050 object
device = torch.device('cpu')

# Initialize the model architecture (must match trained model)
input_dim = 4  # AccX, AccY, AccZ
num_heads = 4
num_layers = 4
num_classes = 2  # Fall or non-fall
num_epochs = 20
dropout = 0.3812751434586094
hidden_dim = 128
learning_rate = 0.0006692265522994677
max_sequence_length=800

# Define GPIO pin for the buzzer
buzzer_pin = 23

# Set up GPIO mode (only done once, no need to clean up after each call)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# Modified TransformerModel to use Performer
class PerformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=hidden_dim, dropout=dropout):
        super(PerformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Use Performer instead of the regular Transformer encoder
        self.performer = Performer(
            dim=hidden_dim,          # Hidden dimension
            depth=num_layers,        # Number of layers
            heads=num_heads,         # Number of attention heads
            dim_head=hidden_dim // num_heads,  # Dimension of each attention head
            causal=False,            # Set to True if you want causal self-attention
            ff_dropout=dropout,      # Feed-forward dropout
            attn_dropout=dropout     # Attention dropout
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.performer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Initialize Performer model
model = PerformerModel(input_dim, num_heads, num_layers, num_classes).to(device)

# Load the model weights
model.load_state_dict(torch.load('/home/ivanursul/performer_model.pt', map_location=device))

# Ensure the model is in evaluation mode
model.eval()


# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F

bus = smbus.SMBus(1)  # I2C bus number on Raspberry Pi


def read_raw_data(addr):
    # Reads the raw data from the specified address (high byte and low byte)
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    # Combine high and low bytes
    value = (high << 8) | low
    # Convert to signed value if needed
    if value > 32768:
        value = value - 65536
    return value


def initialize_mpu(accel_range):
    # Wake up the MPU6050 as it starts in sleep mode
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00)

    # Set the accelerometer range
    if accel_range == 2:
        accel_config_value = 0x00  # ±2g
        scaling_factor = 16384.0
    elif accel_range == 4:
        accel_config_value = 0x08  # ±4g
        scaling_factor = 8192.0
    elif accel_range == 8:
        accel_config_value = 0x10  # ±8g
        scaling_factor = 4096.0
    elif accel_range == 16:
        accel_config_value = 0x18  # ±16g
        scaling_factor = 2048.0
    else:
        raise ValueError("Invalid accelerometer range. Choose from 2, 4, 8, 16.")

    bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, accel_config_value)

    return scaling_factor


def load_offsets(filename):
    with open(filename, 'r') as f:
        offsets = json.load(f)
    logger.info(f"Offsets loaded from {filename}")
    return offsets["acc_x_offset"], offsets["acc_y_offset"], offsets["acc_z_offset"]


acc_x_offset, acc_y_offset, acc_z_offset = load_offsets("/home/ivanursul/mpu_offsets.json")


def read_accelerometer(scaling_factor):
    # Reading the raw accelerometer values
    acc_x = read_raw_data(ACCEL_XOUT_H) - acc_x_offset
    acc_y = read_raw_data(ACCEL_YOUT_H) - acc_y_offset
    acc_z = read_raw_data(ACCEL_ZOUT_H) - acc_z_offset

    # Scale the raw accelerometer values to g-units based on the selected range
    acc_x_g = acc_x / scaling_factor
    acc_y_g = acc_y / scaling_factor
    acc_z_g = acc_z / scaling_factor

    return acc_x_g, acc_y_g, acc_z_g


# Function to calculate the magnitude of a vector (x, y, z)
def calculate_magnitude(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


accel_range = 16
scaling_factor = initialize_mpu(accel_range)
# scaler = StandardScaler()

def calculate_all_spikes(df, spike_threshold, magnitude_field):
    # Initialize variables
    spike_regions = []
    in_spike = False
    spike_start = None

    # Identify where the magnitude crosses the threshold and track spikes
    for i in range(1, len(df)):
        if df[magnitude_field].iloc[i] > spike_threshold and not in_spike:
            # Spike starts
            spike_start = i
            in_spike = True
        elif df[magnitude_field].iloc[i] <= spike_threshold and in_spike:
            # Spike ends
            spike_end = i
            width = spike_end - spike_start

            # Calculate the height of the spike (difference between max and min values during the spike)
            spike_data = df[magnitude_field].iloc[spike_start:spike_end]
            height = spike_data.max()

            # Filter based on width <= 2 and height >= 2
            # if width <= 2 and height >= 2.0:
            #     spike_regions.append((spike_start, spike_end, width, height))

            spike_regions.append((spike_start, spike_end, width, height))

            in_spike = False

    return spike_regions


def apply_linear_interpolation_for_spikes(df, spike_regions, acc_x_column, acc_y_column, acc_z_column):
    # Remove the spike regions and interpolate the values for AccX, AccY, and AccZ
    for spike_start, spike_end, width, height in spike_regions:
        # Set the spike region to NaN for all AccX(g), AccY(g), and AccZ(g)
        df.loc[spike_start:spike_end, [acc_x_column, acc_y_column, acc_z_column]] = np.nan

        # Perform linear interpolation for AccX(g), AccY(g), and AccZ(g)
        df[acc_x_column] = df[acc_x_column].interpolate(method='linear')
        df[acc_y_column] = df[acc_y_column].interpolate(method='linear')
        df[acc_z_column] = df[acc_z_column].interpolate(method='linear')

    # Fill any remaining NaN values using forward or backward filling (no inplace)
    df.fillna({acc_x_column: df[acc_x_column].ffill(),
               acc_y_column: df[acc_y_column].ffill(),
               acc_z_column: df[acc_z_column].bfill()}, inplace=True)

    return df


def detect_multiple_spikes(df, window_size=10, density_threshold=0.5):
    # Calculate a rolling window to detect dense segments of high magnitude
    df['rolling_sum'] = df['Magnitude'].rolling(window=window_size).sum()

    spikes = []
    in_spike = False
    spike_start = None

    for i in range(len(df)):
        if df['rolling_sum'].iloc[i] > density_threshold and not in_spike:
            # Start of a spike
            spike_start = i
            in_spike = True
        elif df['rolling_sum'].iloc[i] <= density_threshold and in_spike:
            # End of the spike
            spike_end = i
            spikes.append((spike_start, spike_end))
            in_spike = False

    # If the spike ends at the last data point
    if in_spike:
        spikes.append((spike_start, len(df) - 1))

    return spikes


def expand_fall_spike_for_multiple_spikes(df, spikes, threshold_factor=0.3):
    expanded_spikes = []

    for (fall_start, fall_end) in spikes:
        # Get the peak value within the current spike
        peak_value = df['Magnitude'].iloc[fall_start:fall_end].max()
        min_threshold = peak_value * threshold_factor

        # Extend backwards
        while fall_start > 0 and df['Magnitude'].iloc[fall_start] > min_threshold:
            fall_start -= 1

        # Extend forwards
        while fall_end < len(df) - 1 and df['Magnitude'].iloc[fall_end] > min_threshold:
            fall_end += 1

        # Add the expanded spike to the list
        expanded_spikes.append((fall_start, fall_end))

    return expanded_spikes


def apply_butter_lowpass_filter_for_non_fall_segments(df, spikes, cutoff=5, fs=100, order=4):
    # Initialize filtered columns with original data
    df['acc_x_filtered'] = df['AccX']
    df['acc_y_filtered'] = df['AccY']
    df['acc_z_filtered'] = df['AccZ']

    # Get all non-spike regions
    last_end = 0
    for spike_start, spike_end in spikes:
        # Filter data before the spike
        if last_end < spike_start:
            df.loc[last_end:spike_start - 1, 'acc_x_filtered'] = butter_lowpass_filter(
                df['AccX'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'acc_y_filtered'] = butter_lowpass_filter(
                df['AccY'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'acc_z_filtered'] = butter_lowpass_filter(
                df['AccZ'].iloc[last_end:spike_start], cutoff, fs, order)

        # After each spike, set the last end to the spike end
        last_end = spike_end + 1

    # Filter the remaining data after the last spike
    if last_end < len(df):
        df.loc[last_end:, 'acc_x_filtered'] = butter_lowpass_filter(df['AccX'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'acc_y_filtered'] = butter_lowpass_filter(df['AccY'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'acc_z_filtered'] = butter_lowpass_filter(df['AccZ'].iloc[last_end:], cutoff, fs, order)

    return df

# Define a Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    padlen = order * 3  # Minimum length required for filtfilt

    if len(data) < padlen:
        # If the data is too short, skip filtering or use an alternative filter
        logger.warning(f"Segment too short for filtfilt (length: {len(data)}), using original data")
        return data  # Optionally, apply a simple moving average filter here instead

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    try:
        # Attempt to apply filtfilt, ensuring padlen condition is met
        y = filtfilt(b, a, data)
    except ValueError as e:
        logger.error(f"Error during filtering: {e}")
        return data  # Return the original data in case of any errors

    return y

def filter_data(df):
    cutoff = 5
    fs = 100
    order = 4
    window_size = 10
    fall_width_threshold = 27
    height_threshold = 2.0

    df['Magnitude'] = np.sqrt(df['AccX'] ** 2 + df['AccY'] ** 2 + df['AccZ'] ** 2)

    # Calculate spikes for non-filtered data
    spike_regions_raw_data = calculate_all_spikes(df, height_threshold, 'Magnitude')

    # Find all spikes that exceed certain width and height
    filtered_spike_regions = [
        (spike_start, spike_end, width, height)
        for spike_start, spike_end, width, height in spike_regions_raw_data
        if width <= 2 and height >= 2.0
    ]

    # Smooth spikes out
    apply_linear_interpolation_for_spikes(df, filtered_spike_regions, 'AccX', 'AccY', 'AccZ')

    fall_spikes = detect_multiple_spikes(df, window_size, fall_width_threshold)
    fall_spikes = expand_fall_spike_for_multiple_spikes(df, fall_spikes, threshold_factor=0.3)
    df = apply_butter_lowpass_filter_for_non_fall_segments(df, fall_spikes, cutoff, fs)

    df['acc_z_filtered'] = df['acc_z_filtered'] - 1

    # Calculate the magnitude of the accelerometer (AccX, AccY, AccZ)
    df['acc_magnitude_filtered'] = np.sqrt(
        df['acc_x_filtered'] ** 2 + df['acc_y_filtered'] ** 2 + df['acc_z_filtered'] ** 2
    )

    return df


scaler = StandardScaler()

# Cycle counter for logging
cycle_count = 0


def play_alarm(beep_count=5, beep_duration=0.2, pause_duration=0.1):
    """
    Play a repetitive beep alarm for fall detection.

    Parameters:
    beep_count (int): Number of beeps in the alarm sequence.
    beep_duration (float): Duration of each beep in seconds.
    pause_duration (float): Pause between each beep in seconds.
    """
    for _ in range(beep_count):
        p = GPIO.PWM(buzzer_pin, 1000)  # 1000Hz frequency for the alarm tone
        p.start(50)  # 50% duty cycle for consistent sound
        time.sleep(beep_duration)  # Duration of the beep
        p.stop()  # Stop the PWM after the beep
        time.sleep(pause_duration)  # Short pause between beeps


def signal_app_start(beep_count=3, beep_duration=0.1, pause_duration=0.05):
    """
    Signal the start of the application by playing a distinct beep sequence.

    Parameters:
    beep_count (int): Number of beeps in the start signal sequence.
    beep_duration (float): Duration of each beep in seconds.
    pause_duration (float): Pause between each beep in seconds.
    """
    for _ in range(beep_count):
        p = GPIO.PWM(buzzer_pin, 2000)  # 2000Hz frequency for start signal
        p.start(50)  # 50% duty cycle for consistent sound
        time.sleep(beep_duration)  # Duration of the beep
        p.stop()  # Stop the PWM after the beep
        time.sleep(pause_duration)  # Short pause between beeps


def collect_and_check_falls():
    global cycle_count
    cycle_count += 1

    data_records = collect_interval_records()

    # Get max magnitude before filtering
    max_magnitude = data_records['Magnitude'].max()

    filtered_data = filter_data(data_records)
    extracted_data = filtered_data[['acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered', 'acc_magnitude_filtered']]
    scaled_data = scaler.fit_transform(extracted_data)
    # Convert data_records into a torch tensor
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

    start_time = time.time()
    batch_data = data_tensor.to(device)

    # Get model predictions
    with torch.no_grad():
        output = model(batch_data)
        predicted_class = torch.argmax(output, dim=1).item()

    inference_time_ms = (time.time() - start_time) * 1000
    result = "FALL" if predicted_class == 1 else "ADL"

    # Consolidated cycle summary log
    logger.info(f"Cycle {cycle_count}: {result} | inference: {inference_time_ms:.1f}ms | max_accel: {max_magnitude:.2f}g")

    if predicted_class == 1:
        logger.warning("FALL DETECTED! Triggering alarm.")
        play_alarm(beep_count=3, beep_duration=0.2, pause_duration=0.2)
        save_csv(filtered_data)


def save_csv(filtered_data):
    # Ensure the folder exists
    folder_path = '/home/ivanursul/accelerometer_data_raw'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Generate a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"accelerometer_data_{timestamp}.csv"
    file_path = os.path.join(folder_path, filename)
    # Convert the data_records to a DataFrame for saving to CSV
    filtered_data.to_csv(file_path, index=False)

def collect_interval_records():
    data_records = pd.DataFrame(
        0.0, index=np.arange(800),
        columns=['AccX', 'AccY', 'AccZ', 'Magnitude']
    )

    # Collect 800 records at 100Hz
    start_time = time.time()
    for i in range(800):
        # Read the sensor data
        acc_x, acc_y, acc_z = read_accelerometer(scaling_factor)

        # Calculate magnitude
        magnitude = calculate_magnitude(acc_x, acc_y, acc_z - 1)

        # Read and filter altitude

        # Assign the data to the DataFrame
        data_records.iloc[i] = [acc_x, acc_y, acc_z - 1, magnitude]

        # Calculate how long the loop took and adjust sleep to maintain 100Hz
        elapsed_time = time.time() - start_time
        expected_time = (i + 1) * 0.01  # 10ms per iteration
        sleep_time = expected_time - elapsed_time

        if sleep_time > 0:
            time.sleep(sleep_time)

    return data_records


signal_app_start()
try:
    while True:
        collect_and_check_falls()
finally:
    GPIO.cleanup()