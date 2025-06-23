import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plots and saves the confusion matrix with AllTrails-inspired styling.

    Parameters:
        y_true (array-like): True class labels
        y_pred (array-like): Predicted class labels
        class_names (list): Class label names (e.g., ['Easy', 'Moderate', 'Hard', 'Very Hard'])
        save_path (str): File name to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    cmap = sns.light_palette("#345C4F", as_cmap=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 12})

    plt.title("Confusion Matrix (FNN Model)", fontsize=14, color="#2E2E2E")
    plt.xlabel("Predicted Label", fontsize=12, color="#2E2E2E")
    plt.ylabel("True Label", fontsize=12, color="#2E2E2E")
    plt.xticks(color="#2E2E2E")
    plt.yticks(color="#2E2E2E")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Confusion matrix saved as: {save_path}")
