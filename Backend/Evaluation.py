from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def Evaluate_Model(true_countries, pred_countries, true_regions, pred_regions):
    accuracy_country = accuracy_score(true_countries, pred_countries)
    precision_country = precision_score(true_countries, pred_countries, average='macro')
    recall_country = recall_score(true_countries, pred_countries, average='macro')
    f1_country = f1_score(true_countries, pred_countries, average='macro')

    accuracy_region = accuracy_score(true_regions, pred_regions)
    precision_region = precision_score(true_regions, pred_regions, average='macro')
    recall_region = recall_score(true_regions, pred_regions, average='macro')
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
