import os
import pandas as pd

# Specify the folder containing the subfolders
base_folder = "/Users/ivanursul/Documents/Projects/Аспірантура/Dataset V4"

# Columns to keep
columns_to_keep = ["AccX", "AccY", "AccZ", "Magnitude", "GyroX", "GyroY", "GyroZ", "Temperature", "Altitude"]

def process_csv_files(base_folder):
    total_files = 0
    removed_files = 0

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):
                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Retain only the specified columns
                    df = df[columns_to_keep]

                    # Check if all records in the 'Altitude' column are empty
                    if df["Altitude"].notna().sum() == 0:
                        # If all values are missing, remove the file
                        os.remove(file_path)
                        removed_files += 1
                        print(f"Removed file: {file_path}")
                    else:
                        # Save the modified CSV
                        df.to_csv(file_path, index=False)
                        print(f"Processed file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    print(f"Final number of CSV files: {total_files - removed_files}")
    print(f"Total files removed: {removed_files}")

# Run the script
process_csv_files(base_folder)
