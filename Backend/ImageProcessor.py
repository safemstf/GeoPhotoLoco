import cv2
import json
# import tensorflow as tf
import numpy as np


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


def save_to_json(data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


img_path = r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\ProcessedImagesVisualized\test.jpg'
processed_tensors = process_image(img_path)
json_path = 'processed_image.json'
save_to_json(processed_tensors, json_path)
