import torch.optim as optim
import logging
import torch
import os

from Net import CNNModel, region_loss_fn, country_loss_fn, distance_loss, ImageDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from config_loader import load_config
from torchvision import transforms
from tqdm import tqdm

# Load configs and start logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = load_config(r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\config.json')  # PC SPECIFIC
NUM_EPOCHS = config['NUM_EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
PROCESSED_IMAGES_FILE = config["TRAIN_IMAGES_FILE"]

print("Parameters Chosen, Initializing...")

# Initialize the model and optimizer
model = CNNModel()
if torch.cuda.is_available():
    model.cuda()  # move to GPU
    print("Using GPU: CUDA AVAILABLE")
else:
    print("not using GPU: CUDA Not Available")

optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])  # learning rates 10 and higher make huge losses.
scheduler = ExponentialLR(optimizer, gamma=config['LR_DECAY'])
loss_fn = distance_loss, country_loss_fn, region_loss_fn

print("Initialized model and optimizer, Loading Data...")

# Data Loading
transform = transforms.Compose([transforms.ToTensor()])  # Include any required transformations
dataset = ImageDataset(PROCESSED_IMAGES_FILE, transform=transform)
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
            images, coords, countries, cities = images.cuda(), coords.cuda(), countries.cuda(), cities.cuda()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # append for evaluation
            true_countries.extend(countries.tolist())
            pred_countries.extend(torch.argmax(country_pred, dim=1).tolist())
            true_regions.extend(cities.tolist())
            pred_regions.extend(torch.argmax(region_pred, dim=1).tolist())

            # Logging evaluation data
            logging.info(f"True Countries: {true_countries}")
            logging.info(f"Predicted Countries: {pred_countries}")
            logging.info(f"True Regions: {true_regions}")
            logging.info(f"Predicted Regions: {pred_regions}")
            logging.info(f"True Coords: {coords}, Predicted Coords: {coord_pred}")
            logging.info(f"Coord Loss: {coord_loss}")
            logging.info(f" Total Loss {total_loss}")

        print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    return true_countries, pred_countries, true_regions, pred_regions
