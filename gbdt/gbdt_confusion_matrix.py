import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# loading the predictions
df = pd.read_csv('model_predictions.csv')

# creating the confusion matrix
y_true = df['true_label']
y_pred = df['predicted_label']
conf_matrix = confusion_matrix(y_true, y_pred)

# plotting the confusion matrix
class_names = ['Easy', 'Medium', 'Hard', 'Very Hard'] 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix_gbdt.png")
plt.show()
