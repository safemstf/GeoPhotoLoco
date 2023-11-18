import csv
import os
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

# Specify the path to your CSV file
csv_file = 'raw2.csv'

# images are down one directory from this file
image_dir = '/home/da0290/Documents/dataExtractor/untagged'  # Update this to your image directory path

output_dir = '/home/da0290/Documents/dataExtractor/tagged'  # Update this to your desired output directory
# os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't already exist

# Initialize empty lists to store the data from each column
keys = []
lons = []
lats = []
cas = []
captured_ats = []

image_paths = []  # To store image file paths

# Read the data from the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    
    # Skip the header row if it exists
    next(csv_reader, None)
    
    # Iterate through the rows and store data in lists
    for row in csv_reader:
        keys.append(row[1])
        lons.append(row[2])
        lats.append(row[3])
        cas.append(row[4])
        captured_ats.append(row[5])
        image_filename = f"{row[1]}.jpg"  # Assuming image filenames correspond to the 'key' values
        image_path = os.path.join(image_dir, image_filename)
        image_paths.append(image_path)
    
# Function to add metadata to an image
def add_metadata_to_image(image_path, metadata_dict, output_path):
    try:
        # Load the existing Exif data
        exif_dict = piexif.load(image_path)
        
        # Merge the existing Exif data with the new metadata
        for key, value in metadata_dict.items():
            # Convert the key to an Exif tag (numeric value)
            exif_tag = piexif.ExifIFD[key]
            exif_dict['Exif'][exif_tag] = str(value)
        
        exif_bytes = piexif.dump(exif_dict)
        
        # Open the image and save with the new metadata
        image = Image.open(image_path)
        image.save(output_path, exif=exif_bytes)
        print(f"Added metadata to {output_path}")
    except Exception as e:
        print(f"Error adding metadata to {image_path}: {e}")



# Add metadata to images and save the modified images
# Create a metadata dictionary for each image
for i in range(len(image_paths)):
    try:
        image_path = image_paths[i]
        metadata_dict = {
            '0th': {
                piexif.ImageIFD.ImageDescription: f"Key: {keys[i]}\nLongitude: {lons[i]}\nLatitude: {lats[i]}\nCA: {cas[i]}\nCaptured At: {captured_ats[i]}"
            }
        }

        exif_bytes = piexif.dump(metadata_dict)
        image = Image.open(image_path)

        # Save the modified image with the new metadata
        image.save(os.path.join(output_dir, f"{keys[i]}_with_metadata.jpg"), exif=exif_bytes)
        print(f"Added metadata to {image_path}")
    except Exception as e:
        print(f"Error adding metadata to {image_path}: {e}")
    

# Add metadata to images and save the modified images
for i in range(len(image_paths)):
    print("Adding metadata to image...")
    output_filename = f"{keys[i]}_with_metadata.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # Load the existing Exif data
    exif_dict = piexif.load(image_paths[i])
    
    # Create the metadata to be added
    metadata = {
        '0th': {
            piexif.ImageIFD.ImageDescription: f"Key: {keys[i]}\nLongitude: {lons[i]}\nLatitude: {lats[i]}\nCA: {cas[i]}\nCaptured At: {captured_ats[i]}"
        }
    }
    print("Metadata created", metadata)
    
    # Merge the existing Exif data with the new metadata
    exif_dict.update(metadata)
    
    # Dump the Exif data to bytes and save the modified image
    exif_bytes = piexif.dump(exif_dict)
    
    # Open the image and save with the new metadata
    image = Image.open(image_paths[i])
    image.save(output_path, exif=exif_bytes)
    print(f"Added metadata to {output_path}")
    print(i)

