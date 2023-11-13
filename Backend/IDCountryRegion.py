import pandas as pd

# Load the CSV file
df = pd.read_csv('BBoxCountriesCities.csv')

# Initialize dictionaries for countries and regions
country_to_idx = {}
region_to_idx = {}

# Populate the dictionaries
for i, row in df.iterrows():
    country, region = row['country'], row['region']

    if country not in country_to_idx:
        country_to_idx[country] = len(country_to_idx)

    if region not in region_to_idx:
        region_to_idx[region] = len(region_to_idx)

pd.Series(country_to_idx).to_csv('country_indices.csv')
pd.Series(region_to_idx).to_csv('region_indices.csv')
