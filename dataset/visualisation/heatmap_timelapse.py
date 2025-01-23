import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import cv2  # Import OpenCV for video creation

# Set the folder path containing CSV files
folder_path = '/Users/ivanursul/Documents/Dataset V3/Falls'  # Replace with your actual path

# Function to extract datetime from filename
def extract_datetime(filename):
    match = re.search(r'accelerometer_data_(\d{8}_\d{6})\.csv', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    else:
        return None

# Get a sorted list of CSV files based on datetime extracted from filenames
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files_with_datetime = [
    (extract_datetime(f), f) for f in csv_files if extract_datetime(f) is not None
]
# Sort the files by datetime
sorted_files = [f for dt, f in sorted(csv_files_with_datetime)]

# Process files in batches
batch_size = 30
num_batches = (len(sorted_files) + batch_size - 1) // batch_size  # Ceiling division

# Create a list to store heatmap image filenames
heatmap_filenames = []

# Loop through each batch
for batch_num in range(1, num_batches + 1):
    # Get the files for the current batch
    batch_files = sorted_files[:batch_num * batch_size]

    # Initialize an empty list to hold the data for heatmap
    heatmap_data = []

    # Loop through each CSV file in the batch
    for file_name in batch_files:
        file_path = os.path.join(folder_path, file_name)

        # Skip empty files
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_name}")
            continue

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Limit the DataFrame to 800 records
        df = df[:800]

        # Check if required columns are present
        if {'AccX', 'AccY', 'AccZ', 'Magnitude'}.issubset(df.columns):
            # Group data by 1-second intervals (100 records each second)
            grouped = df['Magnitude'].groupby(df.index // 100)

            # Store the processed data for each second in a list
            file_data = []

            for _, group in grouped:
                # Divide each 1-second group into 10 intervals of 100ms (10 records per interval)
                high_magnitude_counts = [
                    sum(group[i:i + 10] > 2) for i in range(0, len(group), 10)
                ]
                file_data.append(high_magnitude_counts)

            # Ensure each 'file_data' is converted to a NumPy array and has a consistent shape
            file_data = np.array(file_data)
            heatmap_data.append(file_data)
        else:
            print(f"Required columns missing in {file_name}, skipping.")

    # Skip if no data collected in this batch
    if not heatmap_data:
        print(f"No data collected in batch {batch_num}, skipping heatmap generation.")
        continue

    # Convert the list to a 3D NumPy array (files, seconds, intervals) and average over files
    heatmap_array = np.stack(heatmap_data, axis=0)
    average_heatmap = np.mean(heatmap_array, axis=0)

    # Transpose for plotting
    average_heatmap = average_heatmap.T

    # Find the minimum value and its positions in the average heatmap
    # Flatten the heatmap and get the indices of the top 10 lowest values
    flat_indices = np.argsort(average_heatmap, axis=None)[:10]  # Get indices of 10 smallest values
    top_10_positions = np.unravel_index(flat_indices, average_heatmap.shape)  # Convert to 2D positions
    top_10_values = average_heatmap[top_10_positions]

    # Print the seconds and milliseconds for each of the 10 lowest values in average_heatmap
    print(f"Batch {batch_num}: Top 10 lowest values found at:")
    for idx in range(10):
        pos = (top_10_positions[0][idx], top_10_positions[1][idx])
        second = pos[1] + 1  # 1-based indexing for seconds
        millisecond = pos[0] * 100  # Convert interval index to milliseconds
        print(f"  Value: {top_10_values[idx]} at Second: {second}s, Millisecond: {millisecond}ms")

    # Plot the heatmap with ascending Y-axis labels and white-blue palette
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        np.flipud(average_heatmap),   # Flip the array to make Y-axis ascending
        cmap="plasma",  # Set to white-blue color scheme
        cbar_kws={'label': 'Count'},
        yticklabels=['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'],
        xticklabels=[str(i) for i in range(1, 9)]  # Label X-axis for 8 seconds only
    )
    plt.xlabel("Second")
    plt.ylabel("100 ms intervals")
    plt.title(f"Heatmap of High Magnitude Counts Across Time\nBatch {batch_num * batch_size} Files")
    # Save the heatmap image
    heatmap_filename = f'images/heatmap_batch_{batch_num}.png'
    plt.savefig(heatmap_filename)
    plt.close()
    print(f"Heatmap for batch {batch_num} saved as '{heatmap_filename}'.")
    heatmap_filenames.append(heatmap_filename)

# Create a timelapse video from the saved heatmap images using OpenCV
if heatmap_filenames:
    # Get the dimensions of the images
    frame = cv2.imread(heatmap_filenames[0])
    height, width, layers = frame.shape

    output_dir = "/Users/ivanursul/Downloads"  # Replace with your desired path
    video_name = os.path.join(output_dir, 'timelapse_video.mp4')

    # Use 'mp4v' codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))

    for image_file in heatmap_filenames:
        image = cv2.imread(image_file)
        video.write(image)

    video.release()
    print(f"Timelapse video saved as '{video_name}'.")
else:
    print("No heatmap images were generated, so no timelapse video was created.")

print("All heatmaps have been generated and timelapse video created.")
