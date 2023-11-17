import logging
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config_loader import load_config
from Net import country_to_idx, region_to_idx

config = load_config(r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\config.json')    # PC SPECIFIC


def Evaluate_Model(true_countries, pred_countries, true_regions, pred_regions):
    accuracy_country = accuracy_score(true_countries, pred_countries)
    precision_country = precision_score(true_countries, pred_countries, average='macro', zero_division=0)
    recall_country = recall_score(true_countries, pred_countries, average='macro', zero_division=0)
    f1_country = f1_score(true_countries, pred_countries, average='macro')

    accuracy_region = accuracy_score(true_regions, pred_regions)
    precision_region = precision_score(true_regions, pred_regions, average='macro', zero_division=0)
    recall_region = recall_score(true_regions, pred_regions, average='macro', zero_division=0)
    f1_region = f1_score(true_regions, pred_regions, average='macro')

    # Confusion matrices
    conf_matrix_country = confusion_matrix(true_countries, pred_countries)
    conf_matrix_region = confusion_matrix(true_regions, pred_regions)

    # Plot confusion matrices
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    sns.heatmap(conf_matrix_country, annot=True)
    plt.title('Country Prediction Confusion Matrix')

    plt.subplot(2, 1, 2)
    sns.heatmap(conf_matrix_region, annot=True)
    plt.title('Region Prediction Confusion Matrix')
    plt.show()

    # Print metrics
    print(
        f"Country Accuracy: {accuracy_country}, Precision: {precision_country}, Recall: {recall_country}, F1 Score: {f1_country}")
    print(
        f"Region Accuracy: {accuracy_region}, Precision: {precision_region}, Recall: {recall_region}, F1 Score: {f1_region}")

    # Logging metrics
    logging.info(
        f"Country Accuracy: {accuracy_country}, Precision: {precision_country}, Recall: {recall_country}, F1 Score: {f1_country}")
    logging.info(
        f"Region Accuracy: {accuracy_region}, Precision: {precision_region}, Recall: {recall_region}, F1 Score: {f1_region}")

    # Create a directory for confusion matrix images if it doesn't exist
    os.makedirs('confusion_matrices', exist_ok=True)

    # File names for the confusion matrices
    batch_id = config["PROCESSED_IMAGES_FILE"]
    country_matrix_filename = f'confusion_matrices/country_confusion_matrix_{batch_id}.png'
    region_matrix_filename = f'confusion_matrices/region_confusion_matrix_{batch_id}.png'

    country_labels = [country for country, idx in country_to_idx.items()]
    region_labels = [region for region, idx in region_to_idx.items()]

    # Plot and save country confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_country, annot=True, fmt='g', xticklabels=country_labels, yticklabels=country_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Country Prediction Confusion Matrix - Read as Row: Actual Class, Column: Predicted Class')
    plt.savefig(country_matrix_filename)

    # Plot and save region confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_region, annot=True, fmt='g', xticklabels=region_labels, yticklabels=region_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Region Prediction Confusion Matrix - Read as Row: Actual Class, Column: Predicted Class')
    plt.savefig(region_matrix_filename)
