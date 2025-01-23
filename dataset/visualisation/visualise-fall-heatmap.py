import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# Set the folder path containing CSV files
folder_path = '/Users/ivanursul/Documents/Dataset V4/Falls'  # Replace with your actual path

# Initialize an empty list to hold the data for heatmap
heatmap_data = []

# Loop through each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
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

# Convert the list to a 3D NumPy array (files, seconds, intervals) and average over files
heatmap_array = np.stack(heatmap_data, axis=0)
average_heatmap = np.mean(heatmap_array, axis=0)

# Assuming `heatmap_array` is already computed as in the previous code
# Convert heatmap_array to a list to make it JSON-serializable
heatmap_list = heatmap_array.tolist()

# Get the current working directory
current_directory = os.getcwd()

# Define the file path in the current working directory
json_file_path = os.path.join(current_directory, 'heatmap_array.json')

# Serialize and save to JSON
with open(json_file_path, 'w') as json_file:
    json.dump(heatmap_list, json_file)

print(f"heatmap_array has been serialized to {json_file_path}")

average_heatmap = average_heatmap.T

# Find the minimum value and its positions in the average heatmap
min_value = average_heatmap.min()
min_positions = np.argwhere(average_heatmap == min_value)

# Print the seconds and milliseconds for each lowest value in average_heatmap
print(f"Lowest value ({min_value}) found at:")
for pos in min_positions:
    second = pos[1]  # 1-based indexing for seconds
    millisecond = pos[0] * 100  # Convert interval index to milliseconds
    print(f"  Second: {second}s, Millisecond: {millisecond}ms")

# proceed here: https://chatgpt.com/c/671f942c-7d84-8006-9131-3c2a081d7d30

# Plot the heatmap with ascending Y-axis labels and white-blue palette
plt.figure(figsize=(12, 6))
sns.heatmap(
    np.flipud(average_heatmap),   # Flip the array to make Y-axis ascending
    cmap="Blues",  # Set to white-blue color scheme
    cbar_kws={'label': 'Count'},
    yticklabels=['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'],
    xticklabels=[str(i) for i in range(1, 9)]  # Label X-axis for 8 seconds only
)
plt.xlabel("Second")
plt.ylabel("100 ms intervals")
plt.title("Heatmap of High Magnitude Counts Across Time")
plt.show()

