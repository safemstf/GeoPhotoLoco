import cv2
import json
import os
from tqdm import tqdm
import pandas as pd

# import tensorflow as tf
import numpy as np

bbox_df = pd.read_csv('BBoxCountriesCities.csv')
bbox_df.columns = ['Country', 'Region', 'BBox']


def point_in_bbox(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def find_country_region(lat, lon, bbox_df):
    for _, row in bbox_df.iterrows():
        bbox = list(map(float, row['BBox'].strip('"').split(',')))
        if point_in_bbox(lat, lon, bbox):
            return row['Country'], row['Region']
    return 'Unknown', 'Unknown'


# Processes images into RGB Tensors 1 at a time
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {img_path}")
    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    grid_size = 1
    grid_cells = []

    for y in range(0, 128, grid_size):
        for x in range(0, 128, grid_size):
            cell = img_normalized[y:y + grid_size, x:x + grid_size]
            grid_cells.append(cell)

    # Convert grid cells to lists
    tensors = [cell.tolist() for cell in grid_cells]
    return tensors


# loops through a folder of images with a given batch size to create a json file.
# the images from def process_images_in_folder are run through def process_image(img_path)
def process_images_in_folder(folder_path, json_base_path, BatchSize=2000):
    processed_data = {}
    BatchNuber = 1
    index = 0

    # finds the images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for index, filename in enumerate(tqdm(image_files, desc="Processing Images")):
        img_path = os.path.join(folder_path, filename)
        processed_tensors = process_image(img_path)

        # Extract coordinates from filename (may need to change with different data files)
        lat_lon = filename.rsplit('.', 1)[0]
        lat, lon = map(float, lat_lon.split(','))
        country, region = find_country_region(lat, lon, bbox_df)

        processed_data[f'id_{index}'] = {
            'location': {
                'coord': [lat, lon],
                'country': country,
                'region': region
            },
            'tensors': processed_tensors
        }
        index += 1

        # check if batch is full
        if index % BatchSize == 0 or filename == image_files[-1]:
            json_path = f'{json_base_path}_{BatchNuber}.json'
            save_to_json(processed_data, json_path)
            processed_data = {}
            BatchNuber += 1


def save_to_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


folder_path = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\streetviews'
json_base_path = 'processed_images_batch.json'

process_images_in_folder(folder_path, json_base_path)
