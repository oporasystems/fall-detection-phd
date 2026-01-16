import time
import math
import os
import pandas as pd
import smbus
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import json
from scipy.signal import butter, filtfilt
import RPi.GPIO as GPIO
import board
import busio
import adafruit_bmp3xx
import threading
import requests
import uuid
from logging_config import setup_logging

# Set up logging
logger = setup_logging("fall-collector")

# Define GPIO pin for the buzzer
buzzer_pin = 23
turn_tracking_on_button_pin = 24
voice_on_off_pin = 25

# Set up GPIO mode (only done once, no need to clean up after each call)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

GPIO.setmode(GPIO.BCM)
GPIO.setup(turn_tracking_on_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(voice_on_off_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47
TEMP_OUT_H = 0x41

bus = smbus.SMBus(1)  # I2C bus number on Raspberry Pi

# Create I2C bus at standard speed
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)


def init_bmp_388():
    bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)
    # Set the oversampling for pressure and temperature
    bmp.pressure_oversampling = 8  # Options: 1, 2, 4, 8, 16, 32
    bmp.temperature_oversampling = 8  # Options: 1, 2, 4, 8, 16, 32
    return bmp


# Create sensor object, using the I2C bus
bmp = init_bmp_388()

# Initialize variables for filtering
current_altitude = None
alpha = 0.1  # Smoothing factor for low-pass filter

# Define the sampling rate and corresponding sleep time
sampling_rate = 100  # Hz

window_size = 50  # Define the window size (number of readings to consider in the moving average)

collection_of_data_enabled = False
alert_on = True


def get_sea_level_pressure(location):
    url = f"https://wttr.in/{location}?format=j1"
    try:
        response = requests.get(url, timeout=5)  # Set a timeout for the request
        response.raise_for_status()  # Raises an exception for HTTP errors
        weather_data = response.json()

        # Extract sea-level pressure from the current weather
        pressure = weather_data['current_condition'][0]['pressure']
    except (requests.RequestException, KeyError):
        # If there is no internet or another error, return the default value
        pressure = 1033

    return pressure


# Set accurate sea level pressure
sea_level_pressure = get_sea_level_pressure('Lviv')


def read_altitude_continuously():
    global current_altitude
    while True:
        altitude = bmp.altitude  # Read altitude from BMP388
        current_altitude = altitude  # Update the global variable
        time.sleep(1.0 / sampling_rate)  # Adjust sleep to control reading frequency


# Start the thread for reading altitude
altitude_thread = threading.Thread(target=read_altitude_continuously)
altitude_thread.daemon = True
altitude_thread.start()


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


def calculate_altitude(pressure, sea_level_pressure=1013.25):
    """
    Calculate altitude from pressure using the barometric formula.
    :param pressure: Current pressure in hPa
    :param sea_level_pressure: Sea level standard atmospheric pressure in hPa (default: 1013.25)
    :return: Altitude in meters
    """
    altitude = 44330.0 * (1.0 - (pressure / sea_level_pressure) ** 0.1903)
    return altitude


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


def load_heatmap_array(json_path='/home/ivanursul/heatmap_array.json'):
    try:
        # Load the JSON data
        with open(json_path, 'r') as json_file:
            heatmap_list = json.load(json_file)

        # Convert the list back to a NumPy array
        heatmap_array = np.array(heatmap_list)

        logger.info("Heatmap array loaded successfully.")
        return heatmap_array

    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
    except json.JSONDecodeError:
        logger.error("Error decoding JSON. Please ensure the file is in valid JSON format.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    # Return an empty NumPy array if any error occurs
    return np.empty((0, 8, 10))


def dump_heatmap_array(json_path='/home/ivanursul/heatmap_array.json'):
    global heatmap_array
    heatmap_list = heatmap_array.tolist()

    # Measure the time it takes to save the file
    start_time = time.time()

    with open(json_path, 'w') as json_file:
        json.dump(heatmap_list, json_file)

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Time taken to save the file: {elapsed_time:.4f} seconds")


def find_lowest_value_in_heatmap():
    global heatmap_array

    if heatmap_array.size == 0:
        # Generate default random values
        second = np.random.randint(1, 8)  # Random second between 1 and 7
        millisecond = np.random.choice([i * 100 for i in range(10)])  # Every 100ms from 0 to 900
        index = (second * 100) + (millisecond / 10)
        formatted_time = f"{second}.{millisecond:03d}"

        logger.warning("Heatmap array is empty. Using random default values.")
        logger.info(f"second={second}, millisecond={millisecond}, index={index}, formatted_time={formatted_time}")

        return {
            'lowest_value': 0,
            'second': second,
            'millisecond': millisecond,
            'index': index,
            'formatted': formatted_time
        }

    average_heatmap = np.mean(heatmap_array, axis=0)
    # Transpose the average heatmap if needed
    average_heatmap = average_heatmap.T

    # Find the minimum value and its position(s) in the average heatmap
    min_value = average_heatmap.min()
    min_positions = np.argwhere(average_heatmap == min_value)

    min_positions = np.random.permutation(min_positions)

    # For simplicity, return only the first occurrence of the minimum value
    min_position = min_positions[0]
    second = min_position[1]  # 1-based indexing for seconds
    millisecond = min_position[0] * 100  # Convert interval index to milliseconds
    index = (second * 100) + (millisecond / 10)
    formatted_time = f"{second}.{millisecond:03d}"

    logger.info(f"min_position={min_position}, second={second}, millisecond={millisecond}, index={index}, formatted_time={formatted_time}")

    # Return the lowest value and its position as a dictionary
    return {
        'lowest_value': min_value,
        'second': second,
        'millisecond': millisecond,
        'index': index,
        'formatted': formatted_time
    }


heatmap_array = load_heatmap_array()


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

def read_gyroscope(scaling_factor):
    # Reading the raw gyroscope values
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)

    # Scale the raw gyroscope values to degrees/sec based on the selected range
    gyro_x_dps = gyro_x / scaling_factor
    gyro_y_dps = gyro_y / scaling_factor
    gyro_z_dps = gyro_z / scaling_factor

    return gyro_x_dps, gyro_y_dps, gyro_z_dps


def read_temperature():
    # Read the raw temperature value
    temp_raw = read_raw_data(TEMP_OUT_H)
    # Convert the raw temperature value to Celsius
    temp_c = (temp_raw / 340.0) + 36.53
    return temp_c

# Function to calculate the magnitude of a vector (x, y, z)
def calculate_magnitude(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


accel_range = 16
scaling_factor = initialize_mpu(accel_range)


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

    df['altitude_filtered'] = butter_lowpass_filter(df['Altitude'], cutoff, fs, 4)

    df['acc_z_filtered'] = df['acc_z_filtered'] - 1

    # Calculate the magnitude of the accelerometer (AccX, AccY, AccZ)
    df['acc_magnitude_filtered'] = np.sqrt(
        df['acc_x_filtered'] ** 2 + df['acc_y_filtered'] ** 2 + df['acc_z_filtered'] ** 2
    )

    return df


scaler = StandardScaler()


def play_alarm(beep_count=5, beep_duration=0.2, pause_duration=0.1):
    global alert_on
    if not alert_on:
        return
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


def play_one_beep(beep_duration=0.2):
    global alert_on
    if not alert_on:
        return
    p = GPIO.PWM(buzzer_pin, 500)  # 1000Hz frequency for the alarm tone
    p.start(70)  # 50% duty cycle for consistent sound
    time.sleep(beep_duration)  # Duration of the beep
    p.stop()  # Stop the PWM after the beep


def signal_app_start(beep_count=3, beep_duration=0.1, pause_duration=0.05):
    global alert_on
    if not alert_on:
        return
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


def is_turn_on_off_button_pressed():
    global turn_tracking_on_button_pin
    input_state = GPIO.input(turn_tracking_on_button_pin)
    return input_state == GPIO.LOW  # Return True if button is pressed (LOW state)


def is_alert_button_pressed():
    global voice_on_off_pin
    input_state = GPIO.input(voice_on_off_pin)
    return input_state == GPIO.LOW  # Return True if button is pressed (LOW state)


def sleep(seconds, frequency_hz):
    interval = 1.0 / frequency_hz  # Calculate the interval based on the frequency
    end_time = time.time() + seconds  # Set the target end time 5 seconds from now

    while time.time() < end_time:
        # Sleep for the calculated interval
        time.sleep(interval)

        if is_turn_on_off_button_pressed():
            flip_collect_property()

        if is_alert_button_pressed():
            flip_alert_property()
            time.sleep(1)

    logger.debug(f"Finished sleeping for {seconds} seconds")


def flip_collect_property():
    global collection_of_data_enabled
    collection_of_data_enabled = not collection_of_data_enabled

    if collection_of_data_enabled:
        logger.info("Collection of data was enabled")
    else:
        logger.info("Collection of data was disabled")

    play_alarm(beep_count=3, beep_duration=0.2, pause_duration=0.2)


def flip_alert_property():
    global alert_on
    alert_on = not alert_on

    if alert_on:
        logger.info("Alert buzzer is on")
        play_alarm(beep_count=5, beep_duration=0.2, pause_duration=0.2)
    else:
        logger.info("Alert buzzer is off")


def add_high_magnitude_counts(data_records):
    global heatmap_array
    # Limit the DataFrame to 800 records
    data_records = data_records[:800]

    # Check if required columns are present
    if {'AccX', 'AccY', 'AccZ', 'Magnitude'}.issubset(data_records.columns):
        # Group data by 1-second intervals (100 records each second)
        grouped = data_records['Magnitude'].groupby(data_records.index // 100)

        # Store the processed data for each second in a list
        file_data = []

        for _, group in grouped:
            # Divide each 1-second group into 10 intervals of 100ms (10 records per interval)
            high_magnitude_counts = [
                sum(group[i:i + 10] > 2) for i in range(0, len(group), 10)
            ]
            file_data.append(high_magnitude_counts)

        # Convert file_data to a NumPy array with shape (seconds, intervals)
        file_data = np.array(file_data)

        # Check if file_data has any high magnitude counts (values > 0)
        if np.any(file_data > 0):
            logger.info(f"Processed file_data with high magnitude counts, shape {file_data.shape}:")
            logger.debug(file_data)
            heatmap_array = np.vstack([heatmap_array, file_data[np.newaxis, ...]])
            logger.info(f"Updated heatmap_array shape: {heatmap_array.shape}")
        else:
            logger.info("No high magnitude counts found in file_data. Skipping addition to heatmap_array.")

    else:
        raise ValueError("DataFrame is missing required columns: 'AccX', 'AccY', 'AccZ', 'Magnitude'")


def collect():
    global collection_of_data_enabled, bmp

    if collection_of_data_enabled:
        logger.info("Iteration started")
        lowest_value = find_lowest_value_in_heatmap()
        index_of_interest = lowest_value['index']

        play_alarm(beep_count=1, beep_duration=0.2, pause_duration=0.2)

        data_records = collect_interval_records(index_of_interest)

        if not collection_of_data_enabled:
            logger.info("Collection of data was disabled, skipping")
            return

        filtered_data = filter_data(data_records)
        save_csv(filtered_data)

        play_alarm(beep_count=2, beep_duration=0.2, pause_duration=0.2)

        sleep(3, 100)

        add_high_magnitude_counts(data_records)
        # Dumping array to make sure that the next time device starts - the new data will be there.
        dump_heatmap_array()
    else:
        sleep(1, 100)

    if is_turn_on_off_button_pressed():
        flip_collect_property()
        sleep(1, 100)


def save_csv(filtered_data):
    global collection_of_data_enabled
    if not collection_of_data_enabled:
        logger.info("Collection of data was disabled, skipping")
        return

    logger.info("Started saving csv")
    # Define the main folder path
    base_folder_path = '/home/ivanursul/accelerometer_data_raw'
    # Generate the date-based subfolder name
    date_subfolder = time.strftime("%d_%m_%Y")
    folder_path = os.path.join(base_folder_path, date_subfolder)

    # Ensure the subfolder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a unique filename with timestamp and salt
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    salt = uuid.uuid4().hex[:8]  # Generate an 8-character unique salt
    filename = f"accelerometer_data_{timestamp}_{salt}.csv"
    file_path = os.path.join(folder_path, filename)

    # Save the filtered data to the CSV file
    filtered_data.to_csv(file_path, index=False)

    logger.info(f"Data successfully saved to {file_path}")


def collect_interval_records(index_of_interest):
    global current_altitude
    data_records = pd.DataFrame(
        0.0, index=np.arange(800),
        columns=['AccX', 'AccY', 'AccZ', 'Magnitude', 'GyroX', 'GyroY', 'GyroZ', 'Temperature', 'Altitude']
    )

    logger.info(f"Index of interest: {index_of_interest}")

    # Check if the index_of_interest is less than 100, beep and apply delay
    if index_of_interest < 50:
        logger.warning(f"Approaching index of interest for the first second: {index_of_interest}")
        #play_alarm(4, beep_duration=0.02, pause_duration=0.03)
        # Calculate and apply the delay
        delay = 0.7 + (index_of_interest * 0.01)  # 500 ms + index_of_interest * 10 ms
        time.sleep(delay)

    # Collect 800 records at 100Hz
    start_time = time.time()
    for i in range(800):
        if is_turn_on_off_button_pressed():
            flip_collect_property()
            break

        # Beep if approaching index_of_interest within the loop
        if i == max(0, index_of_interest - 100) and 0 <= index_of_interest <= 800:
            logger.warning(f"Approaching index of interest: {index_of_interest}")
            play_one_beep(beep_duration=0.02)

        # # Check if index_of_interest is greater than 900 and beep 200 records before
        # if i == max(0, index_of_interest - 100) and index_of_interest > 700:
        #     print(f"Warning: High index of interest: {index_of_interest}. Preparing with 2-second delay.")
        #     play_alarm(2, beep_duration=0.02, pause_duration=0.01)

        # Read the sensor data
        acc_x, acc_y, acc_z = read_accelerometer(scaling_factor)
        gyro_x, gyro_y, gyro_z = read_gyroscope(scaling_factor)
        temperature = read_temperature()

        # Calculate magnitude
        magnitude = calculate_magnitude(acc_x, acc_y, acc_z - 1)

        # Read and filter altitude
        altitude = current_altitude

        # Assign the data to the DataFrame
        data_records.iloc[i] = [acc_x, acc_y, acc_z - 1, magnitude, gyro_x, gyro_y, gyro_z, temperature, altitude]

        # Calculate how long the loop took and adjust sleep to maintain 100Hz
        elapsed_time = time.time() - start_time
        expected_time = (i + 1) * 0.01  # 10ms per iteration
        sleep_time = expected_time - elapsed_time

        if sleep_time > 0:
            time.sleep(sleep_time)

    # Measure total execution time
    total_time = time.time() - start_time

    logger.info(f"Total execution time: {total_time:.6f} seconds")

    return data_records


signal_app_start()
try:
    while True:
        collect()
finally:
    GPIO.cleanup()