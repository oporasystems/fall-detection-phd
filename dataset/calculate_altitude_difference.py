import os
import pandas as pd

# Define the directory containing the CSV files
folder_path = '/Users/ivanursul/Documents/Dataset V3/ADL'

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Check if the file is a CSV
    if file_name.endswith(".csv"):
        # Check if the file size is zero
        if os.path.getsize(file_path) == 0:
            print(f"Removing empty file: {file_name}")
            os.remove(file_path)
            continue

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_path)

            # Skip if the DataFrame is empty
            if df.empty:
                print(f"Removing empty DataFrame file: {file_name}")
                os.remove(file_path)
                continue

            # Calculate Altitude delta
            initial_altitude = df['altitude_filtered'].iloc[0]
            df['Altitude_Delta'] = df['altitude_filtered'] - initial_altitude

            # Save the updated DataFrame back to the file
            df.to_csv(file_path, index=False)
            print(f"Processed file: {file_name}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
