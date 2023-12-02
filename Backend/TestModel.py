import cv2
import torch
import os
import json
import time
import pandas as pd
from config_loader import load_config
from Net import CNNModel, country_to_idx, region_to_idx  # Import your CNN model class


current_directory = os.getcwd()

# Construct the path to the target file one directory up
config = os.path.join(os.path.dirname(current_directory), "Backend", "config.json")

config = load_config(config)

# Paths from the config file
modelFile = os.path.join(os.path.dirname(current_directory), "Backend", "model.pth")
testImageFolder = os.path.join(os.path.dirname(current_directory), "Backend", "TestImages")

model_file_path = modelFile
test_images_folder = testImageFolder

idx_to_country = {value: key for key, value in country_to_idx.items()}  # Reverse your existing country_to_idx
idx_to_region = {value: key for key, value in region_to_idx.items()}  # Reverse your existing region_to_idx

BBoxFile = os.path.join(os.path.dirname(current_directory), "Backend", "BBoxCountriesCities.csv")
bbox_df = pd.read_csv(BBoxFile)
bbox_df.columns = ['Country', 'Region', 'BBox']


def get_point_in_bbox(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def get_true_country_region(lat_lon_str, bbox_df):
    try:
        # Attempt to parse the latitude and longitude
        lat, lon = map(float, lat_lon_str.split(','))

        # Check if the point is in any of the bounding boxes
        for _, row in bbox_df.iterrows():
            bbox = list(map(float, row['BBox'].strip('"').split(',')))
            if get_point_in_bbox(lat, lon, bbox):
                return row['Country'], row['Region']

    except ValueError:
        # If the string cannot be converted to floats, return 'Unknown'
        pass

    return 'Unknown', 'Unknown'


# load the trained model
def load_trained_model(path):
    model = CNNModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # Add map_location if using CPU
    model.eval()
    return model


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {img_path}")
    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    return torch.tensor(img_normalized, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)


# find the most recent image in a folder
def find_most_recent_image(folder_path):
    list_of_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    if not list_of_files:  # check if list is empty
        raise FileNotFoundError("No files found in the folder.")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# predict using the model
def predict(model, img_tensor):
    with torch.no_grad():
        country_pred, region_pred, coord_pred = model(img_tensor)

    return country_pred, region_pred, coord_pred

def process_predictions(country_pred, region_pred, coord_pred):
    # Convert indices to actual names
    country_idx = torch.argmax(country_pred, dim=1).item()
    region_idx = torch.argmax(region_pred, dim=1).item()

    country_name = idx_to_country.get(country_idx, "Unknown")
    region_name = idx_to_region.get(region_idx, "Unknown")

    # Convert coordinates to readable format
    coordinates = coord_pred[0].tolist()  # Assuming it's a single prediction

    return {
        "Country": country_name,
        "Region": region_name,
        "Coordinates": coordinates
    }

# Main function to handle model loading and prediction
def main(interval=2):
    model = load_trained_model(model_file_path)
    processed_images = set()
    while True:
        img_path = find_most_recent_image(test_images_folder)
        # Extract true values from the image filename
        filename = os.path.basename(img_path)
        lat_lon = filename.rsplit('.', 1)[0]
        last_processed = None
        try:
            img_path = find_most_recent_image(test_images_folder)
            if img_path != last_processed and img_path not in processed_images:
                processed_images.add(img_path)
                try:
                    true_lat, true_lon = map(float, lat_lon.split(','))
                    true_country, true_region = get_true_country_region(lat_lon, bbox_df)
                except ValueError:
                    # If the filename does not contain valid coordinates, set to Unknown
                    true_lat, true_lon = 'Unknown', 'Unknown'
                    true_country, true_region = 'Unknown', 'Unknown'

                img_tensor = process_image(img_path)
                country_pred, region_pred, coord_pred = predict(model, img_tensor)

                # Process the predictions
                processed_prediction = process_predictions(country_pred, region_pred, coord_pred)

                # Add true values to the output
                processed_prediction['True_Coordinates'] = [true_lat, true_lon] if true_lat != 'Unknown' else 'Unknown'
                processed_prediction['True_Country'] = true_country
                processed_prediction['True_Region'] = true_region

                return processed_prediction

            else:
                print("No new images to process.")
        except FileNotFoundError as e:
            print(str(e))

        time.sleep(interval)


# Example usage
if __name__ == "__main__":
    prediction = main(interval=2)
    print(json.dumps(prediction, indent=4))
