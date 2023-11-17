import re
import pandas as pd

# Load the bounding box data
bbox_df = pd.read_csv('BBoxCountriesCities.csv')
bbox_df.columns = ['Country', 'Region', 'BBox']


def point_in_bbox2(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def find_country_region1(lat, lon, bbox_df):
    for _, row in bbox_df.iterrows():
        bbox = list(map(float, row['BBox'].strip('"').split(',')))
        if point_in_bbox2(lat, lon, bbox):
            return row['Country'], row['Region']
    return 'Unknown', 'Unknown'


def parse_log_file(log_file_path):
    true_coords = []
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if 'True Coords:' in line:
                coords_str = re.search(r"True Coords: tensor\(\[\[([^\]]+)\]\]", line)
                if coords_str:
                    coords_pairs = coords_str.group(1).split('],\n')
                    for pair in coords_pairs:
                        lat, lon = map(float, pair.replace('[', '').replace(']', '').split(', '))
                        true_coords.append((lat, lon))
    return true_coords


# Parse the log file
log_file_path = 'training.log'  # Update the path as needed
true_coords = parse_log_file(log_file_path)

# Match coordinates to countries and regions
matched_countries_regions = [find_country_region1(lat, lon, bbox_df) for lat, lon in true_coords]

# Count occurrences of each country
country_counts = pd.Series([country for country, _ in matched_countries_regions]).value_counts()
print(country_counts)
