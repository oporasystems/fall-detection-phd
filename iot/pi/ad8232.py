import time
import board
import busio
import numpy as np
import scipy.signal as signal
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize ADS1115 ADC at address 0x48
ads = ADS.ADS1115(i2c, address=0x48)

# Set the gain
ads.gain = 1

# Create a single-ended analog input channel on pin A0
channel = AnalogIn(ads, ADS.P0)

# Sampling parameters
SAMPLE_RATE = 250  # Samples per second
WINDOW_DURATION = 5  # Duration of the sliding window in seconds
BUFFER_SIZE = SAMPLE_RATE * WINDOW_DURATION  # Total samples in the buffer

# Initialize data buffer
data_buffer = []

print("Reading AD8232 heart rate data and estimating heart rate every second...")

try:
    while True:
        start_time = time.time()
        # Read data for 1 second
        num_samples = SAMPLE_RATE  # Number of samples to read per second
        for _ in range(num_samples):
            voltage = channel.voltage
            data_buffer.append(voltage)
            # Keep the buffer size within WINDOW_DURATION
            if len(data_buffer) > BUFFER_SIZE:
                data_buffer.pop(0)
            time.sleep(1 / SAMPLE_RATE)

        # Proceed only if we have enough data in the buffer
        if len(data_buffer) == BUFFER_SIZE:
            # Convert data buffer to numpy array
            ecg_signal = np.array(data_buffer)

            # Preprocess the signal
            # Remove DC offset
            ecg_signal = ecg_signal - np.mean(ecg_signal)

            # Apply bandpass filter to remove noise (e.g., 5-15 Hz for ECG)
            fs = SAMPLE_RATE  # Sampling frequency
            lowcut = 5.0
            highcut = 15.0

            # Design the filter
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(1, [low, high], btype='band')

            # Apply the filter
            filtered_ecg = signal.lfilter(b, a, ecg_signal)

            # Detect peaks (QRS complexes)
            # Use scipy's find_peaks function
            peak_height = 0.5 * np.std(filtered_ecg)
            peaks, properties = signal.find_peaks(
                filtered_ecg,
                distance=fs * 0.6,  # Minimum 0.6 seconds between peaks
                height=peak_height
            )

            # Calculate heart rate
            peak_times = peaks / fs  # Convert sample indices to time
            rr_intervals = np.diff(peak_times)  # Time between successive peaks
            if len(rr_intervals) > 0:
                avg_rr_interval = np.mean(rr_intervals)
                heart_rate = 60 / avg_rr_interval
                print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
            else:
                print("No heartbeats detected. Please check sensor connection.")
        else:
            print("Collecting data...")

        # Wait until 1 second has passed
        elapsed_time = time.time() - start_time
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)

except KeyboardInterrupt:
    print("Data reading stopped by user.")