import logging
import pandas as pd
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config_loader import load_config

config = load_config(r'C:\Users\safem\PycharmProjects\GeoPhotoLoco\Backend\config.json')  # PC SPECIFIC

country_indices_df = pd.read_csv('country_indices.csv', header=None)
region_indices_df = pd.read_csv('region_indices.csv', header=None)

idx_to_country = pd.Series(country_indices_df[0].values, index=country_indices_df[1]).to_dict()
idx_to_region = pd.Series(region_indices_df[0].values, index=region_indices_df[1]).to_dict()


def Evaluate_Model(true_countries, pred_countries, true_regions, pred_regions):
    # Map numerical indices to names for countries
    true_country_names = [idx_to_country.get(idx, 'Unknown') for idx in true_countries]
    pred_country_names = [idx_to_country.get(idx, 'Unknown') for idx in pred_countries]

    # Map numerical indices to names for regions (if needed)
    true_region_names = [idx_to_region.get(idx, 'Unknown') for idx in true_regions]
    pred_region_names = [idx_to_region.get(idx, 'Unknown') for idx in pred_regions]

    accuracy_country = accuracy_score(true_country_names, pred_country_names)
    precision_country = precision_score(true_country_names, pred_country_names, average='macro', zero_division=0)
    recall_country = recall_score(true_country_names, pred_country_names, average='macro', zero_division=0)
    f1_country = f1_score(true_country_names, pred_country_names, average='macro')

    accuracy_region = accuracy_score(true_region_names, pred_region_names)
    precision_region = precision_score(true_region_names, pred_region_names, average='macro', zero_division=0)
    recall_region = recall_score(true_region_names, pred_region_names, average='macro', zero_division=0)
    f1_region = f1_score(true_region_names, pred_region_names, average='macro')

    country_labels_used = sorted(set(true_country_names + pred_country_names))
    region_labels_used = sorted(set(true_region_names + pred_region_names))

    # Confusion matrices
    conf_matrix_country = confusion_matrix(true_country_names, pred_country_names, labels=country_labels_used)
    conf_matrix_region = confusion_matrix(true_region_names, pred_region_names, labels=region_labels_used)

    # Plot confusion matrices
    plt.figure(figsize=(30, 30))
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
    batch_id = config["TRAIN_IMAGES_FILE"]
    country_matrix_filename = f'confusion_matrices/country_confusion_matrix_{batch_id}.png'
    region_matrix_filename = f'confusion_matrices/region_confusion_matrix_{batch_id}.png'

    # Plot and save country confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_country, annot=True, fmt='g', xticklabels=country_labels_used, yticklabels=country_labels_used)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Country Prediction Confusion Matrix - Read as Row: Actual Class, Column: Predicted Class')
    plt.tight_layout()
    plt.savefig(country_matrix_filename, dpi=300)

    # Plot and save region confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_region, annot=True, fmt='g', xticklabels=region_labels_used, yticklabels=region_labels_used)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Region Prediction Confusion Matrix - Read as Row: Actual Class, Column: Predicted Class')
    plt.tight_layout()
    plt.savefig(region_matrix_filename, dpi=300)
