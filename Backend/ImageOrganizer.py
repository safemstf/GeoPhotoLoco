import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# Paths to your dataset folders
dataset_paths = [r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\AmsterdamTagged',
                 r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\BudapestTagged',
                 r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\Moscow_Tagged']

# Dictionary to track images from each location
image_paths = defaultdict(list)

# Gather image paths
for dataset_path in dataset_paths:
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc=f'Processing {root}'):
            if file.endswith('.jpg') or file.endswith('.jpeg'):  # Add other image formats if needed
                image_paths[os.path.basename(root)].append(os.path.join(root, file))

# Determine batch size - customize this based on your needs
batch_size = 1000  # Example batch size

# Destination path for organized batches
destination_path = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\OrganizedImageBatches'

# Initialize batch number
batch_number = 1

# Process and organize images into batches
while True:
    batch = []
    for location, paths in image_paths.items():
        if len(paths) < batch_size // len(dataset_paths):
            continue

        selected_images = random.sample(paths, batch_size // len(dataset_paths))
        batch.extend(selected_images)

        # Remove selected images to avoid duplication in future batches
        for img in selected_images:
            paths.remove(img)

    if not batch:
        break  # No more images to process

    # Create a batch directory
    batch_dir = os.path.join(destination_path, f'batch_{batch_number}')
    os.makedirs(batch_dir, exist_ok=True)

    # Copy images to the batch directory
    for img_path in tqdm(batch, desc=f'Creating batch_{batch_number}'):
        shutil.copy(img_path, os.path.join(batch_dir, os.path.basename(img_path)))

    batch_number += 1

print("Batch organization complete.")
