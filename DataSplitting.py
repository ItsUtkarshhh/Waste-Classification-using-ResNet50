import os
import shutil
import random
from math import ceil

# Paths
dataset_folder = 'Dataset'  # This is the folder where the dataset is now located
validation_folder = 'Data Validation'  # This is where 10% of files will be copied to

# Create validation folder if it doesn't exist
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# Iterate through each subfolder (class of waste)
for subfolder in os.listdir(dataset_folder):
    subfolder_path = os.path.join(dataset_folder, subfolder)

    # Ensure it's a directory (skip any files)
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        random.shuffle(files)  # Shuffle files for random selection

        # Calculate 10% of files (rounding up)
        validation_count = ceil(0.1 * len(files))
        validation_files = files[:validation_count]

        # Create a corresponding subfolder in the Data Validation folder
        validation_subfolder_path = os.path.join(validation_folder, subfolder)
        if not os.path.exists(validation_subfolder_path):
            os.makedirs(validation_subfolder_path)

        # Copy selected files to the validation folder
        for file in validation_files:
            shutil.move(os.path.join(subfolder_path, file), os.path.join(validation_subfolder_path, file))

print("10% of data has been moved to the Data Validation folder.")