import torch
import torch.nn as nn
import json
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt


# # transform options:
# Basic Transforms:
#
# transforms.Resize((width, height)): Resizes the image to the specified size.
# transforms.CenterCrop(size): Crops the center of the image to the given size.
# transforms.ToTensor(): Converts a PIL image or NumPy ndarray to a PyTorch tensor.

# Augmentation Transforms (useful for increasing dataset variability):

# transforms.RandomHorizontalFlip(): Horizontally flips the image randomly with a given probability.
# transforms.RandomRotation(degrees): Rotates the image by a random angle within the specified range.
# transforms.ColorJitter(): Randomly changes the brightness, contrast, and saturation of an image.

# Composing Transforms:
# Transforms can be composed using
# transforms.Compose([transforms]), where transforms is a list of transformations applied sequentially.

def load_csv_to_dict(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header
        return {rows[0]: int(rows[1]) for rows in reader}


# Load the CSV files
country_to_idx = load_csv_to_dict('country_indices.csv')
region_to_idx = load_csv_to_dict('region_indices.csv')


# dataset
class ImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.data = {}
        for key in tqdm(data.keys(), desc="Loading Data"):
            item = data[key]
            self.data[key] = {
                'tensors': torch.tensor(item['tensors']),
                'coord': torch.tensor(item['location']['coord']),
                'country': country_to_idx.get(item['location']['country'], 253),  # 253 is unknown
                'region': region_to_idx.get(item['location']['region'], 4440)  # 4440 is unknown
            }

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[list(self.data.keys())[idx]]

        image = item['tensors'].permute(2, 0, 1)
        coord = item['coord']
        country = item['country']
        region = item['region']

        return image, coord, country, region


# loss functions for coord, country, and city/region
def country_loss_fn(output, target, unknown_penalty=5):
    loss = nn.CrossEntropyLoss()(output, target)
    # Calculate additional penalty for predicting 'unknown'
    unknown_index = 253  # Index for 'unknown'
    predictions = torch.argmax(output, dim=1)
    penalty_mask = (predictions == unknown_index)
    penalty = torch.tensor(penalty_mask, dtype=torch.float).sum() * unknown_penalty

    # Total loss is the sum of cross-entropy loss and penalty
    total_loss = loss + penalty
    return total_loss


def region_loss_fn(output, target, unknown_penalty=50):
    loss = nn.CrossEntropyLoss()(output, target)
    # Calculate additional penalty for predicting 'unknown'
    unknown_index = 4440  # Index for 'unknown'
    predictions = torch.argmax(output, dim=1)
    penalty_mask = (predictions == unknown_index)
    penalty = torch.tensor(penalty_mask, dtype=torch.float32).sum() * unknown_penalty

    # Total loss is the sum of cross-entropy loss and penalty
    total_loss = loss + penalty
    return total_loss


# distance formula in meters
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # radius of earth in kilometers.
    return c * r


def distance_loss(output, target):
    batch_size = output.shape[0]
    total_loss = 0
    penalty_factor = 45075  # Large value penalize circumference of earth

    for i in range(batch_size):
        pred_lon, pred_lat = output[i][0].item(), output[i][1].item()
        true_lon, true_lat = target[i][0].item(), target[i][1].item()

        # Check for out-of-range coordinates
        if abs(pred_lon) > 180 or abs(pred_lat) > 90:
            total_loss += penalty_factor + haversine(pred_lon, pred_lat, true_lon, true_lat) ** 2
        else:
            total_loss += haversine(pred_lon, pred_lat, true_lon, true_lat)

    return total_loss / (batch_size * 100)


# CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # kernel_size=3 each filter is 3x3. Odd numbers have a central pixel. Good for spatial reference
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc_country = nn.Linear(64 * 32 * 32, 254)  # num_countries
        self.fc_city = nn.Linear(64 * 32 * 32, 4441)  # num_cities/regions
        self.fc_coord = nn.Linear(64 * 32 * 32, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = F.elu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = F.elu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        country_pred = self.fc_country(x)
        region_pred = self.fc_city(x)
        coord_pred = self.fc_coord(x)
        return country_pred, region_pred, coord_pred
