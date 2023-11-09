import json
import mapillary.config.api.entities as entities
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mapillary_data_fetch')

with open('config.json', 'r') as file:
    config = json.load(file)

MLY_ACCESS_TOKEN = config['MLY_ACCESS_TOKEN']
entities.set_access_token(MLY_ACCESS_TOKEN)

SetLimit = 5

# Setting bounding box for the UK
bbox = [-14.564613, 48.875818, 2.614298, 59.827703]

file_name = 'ImageMetaData_in_bbox_UK.json'

# Initialize counter
request_count = 0
request_limit = 5

while request_count < request_limit:
    data = fetch_data()

    # Increment the counter
    request_count += 1

    # Log the request
    logger.info(f'Request {request_count} made')

    # Conditional check to stop making requests if the limit is reached
    if request_count >= request_limit:
        logger.info(f'Request limit of {request_limit} reached')
        break  # Exit the loop


def fetch_data(file_name):
    # Fetch image metadata
    image_metadata = entities.search_for_images(
        bbox=bbox,
        limit=SetLimit,
        image_type='flat'
    )

    # Logging the data fetching operation
    logger.info(f"[search_for_images] Data fetched successfully")

    # Save the data to a file
    with open(file_name, mode="w") as f:
        json.dump(image_metadata, f, indent=4)
        logger.info(f"[search_for_images] Data saved to {file_name}")

    return data


# Fetch and save data
fetch_data(file_name)
