import csv
import os
from PIL import Image

# Specify the path to your CSV file
# PC SPECIFIC
csv_file = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\MetaData\train_val\budapest\database\raw.csv'

# images are down one directory from this file
image_dir = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Budapest_Images'  # Update this to your image directory path

output_dir = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\BudapestTagged'  # Update this to your desired output directory

os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't already exist

# Initialize empty lists to store the data from each column
keys = []
lons = []
lats = []
image_paths = []  # To store image file paths

# Read the data from the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    # Iterate through the rows and store data in lists
    for row in csv_reader:
        keys.append(row[1])
        lons.append(row[2])
        lats.append(row[3])
        image_filename = f"{row[1]}.jpg"  # Assuming image filenames correspond to the 'key' values
        image_path = os.path.join(image_dir, image_filename)
        image_paths.append(image_path)

for i in range(len(image_paths)):
    new_filename = f"{lats[i]},{lons[i]}.jpg"  # Format: 'latitude,longitude.jpg'
    new_path = os.path.join(output_dir, new_filename)

    if os.path.exists(image_paths[i]):
        if not os.path.exists(new_path):  # Check if the new filename already exists
            os.rename(image_paths[i], new_path)
            print(f"Renamed {image_paths[i]} to {new_path}")
        else:
            print(f"Skipping {image_paths[i]}: {new_path} already exists.")
    else:
        print(f"Skipping: {image_paths[i]} does not exist.")

# for i in range(len(image_paths)):
#     print(i)
# print("Renaming image...")
# new_filename = f"{lats[i]},{lons[i]}.jpg"
# print(new_filename)
# new_path = os.path.join(output_dir, new_filename)
# print(new_path)

# # Check if the file exists before attempting to rename
# if os.path.exists(image_paths[i]):
#     os.rename(image_paths[i], new_path)
#     print(f"Renamed {image_paths[i]} to {new_path}")
# else:
#     print(f"Skipping: {image_paths[i]} does not exist.")
# print(i)


# Store as latitude,longitude.jpg
