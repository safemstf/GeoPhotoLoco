import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns
import torch
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from Net import CNNModel, region_loss_fn, country_loss_fn, distance_loss, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Evaluation import Evaluate_Model

# Parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32

print("Parameters Chosen, Initializing...")

# Initialize the model and optimizer
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = distance_loss, country_loss_fn, region_loss_fn

print("Initialized model and optimizer, Loading Data...")

# Data Loading
transform = transforms.Compose([transforms.ToTensor()])  # Include any required transformations
dataset = ImageDataset('processed_images_batch_1.json', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Data Loaded, starting Training...")

# Lists for evaluation data
true_countries, pred_countries = [], []
true_regions, pred_regions = [], []


# Training loop
def TrainGeoPhotoLoco(resume=False):
    if resume and os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
        print("Resuming from saved state.")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for images, coords, countries, cities in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            country_pred, region_pred, coord_pred = model(images)

            # Calculate loss for each task and combine them
            country_loss = country_loss_fn(country_pred, countries)
            region_loss = region_loss_fn(region_pred, cities)
            coord_loss = distance_loss(coord_pred, coords)

            print(f'Distance Loss: {distance_loss(coord_pred, coords)}, '
                  f'Country Loss: {country_loss},'
                  f' region_loss: {region_loss}')

            total_loss = country_loss + region_loss + coord_loss
            total_loss.backward()
            optimizer.step()

            # append for evaluation
            true_countries.extend(countries.tolist())
            pred_countries.extend(torch.argmax(country_pred, dim=1).tolist())
            true_regions.extend(cities.tolist())
            pred_regions.extend(torch.argmax(region_pred, dim=1).tolist())

        print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    TrainGeoPhotoLoco(resume=True)
Evaluate_Model(true_countries, pred_countries, true_regions, pred_regions)
