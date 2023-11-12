import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 10
BATCH_SIZE = 32


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


# dataset
class ImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_key = list(self.data.keys())[idx]
        image = torch.tensor(self.data[image_key])
        country = torch.tensor(item['location']['country'])
        region = torch.tensor(item['location']['region'])

        # Extracting latitude and longitude from the filename
        _, lat_lon = image_key.split('_')
        lat, lon = lat_lon[:-4].split(',')
        coord = torch.tensor([float(lat), float(lon)])

        if self.transform:
            image = self.transform(image)

        return image, coord, country, region


# loss function
# todo: modify the distance loss for better usage and calculation
# todo: hierarchical classification gain/loss
def distance_loss(output, target):
    return torch.sqrt(torch.sum((output - target) ** 2, dim=1)).mean()


# CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc_country = nn.Linear(64 * 32 * 32, 245)  # num_countries
        self.fc_city = nn.Linear(64 * 32 * 32, 4447)  # num_cities/regions
        self.fc_coord = nn.Linear(64 * 32 * 32, 2)

    # todo: see if elu is possible(training time)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = torch.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        country_pred = self.fc_country(x)
        region_pred = self.fc_city(x)
        coord_pred = self.fc_coord(x)
        return country_pred, region_pred, coord_pred


# Load the data
transform = transforms.Compose([transforms.ToTensor()])  # Include any required transformations
dataset = ImageDataset('processed_images_batch.json_1.json', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model, optimizer, and loss function
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = distance_loss

# Training loop
for epoch in range(NUM_EPOCHS):
    for images, coords, countries, cities in dataloader:
        optimizer.zero_grad()
        country_pred, region_pred, coord_pred = model(images)

        # Calculate loss for each task and combine them
        country_loss = country_loss_fn(country_pred, countries)  # todo: Define country loss function
        city_loss = region_loss_fn(region_pred, cities)  # todo: Define region loss function
        coord_loss = distance_loss(coord_pred, coords)
        total_loss = country_loss + city_loss + coord_loss

        total_loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')
