import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading Data
alltrails = pd.read_csv('alltrails-data.csv')
chat = pd.read_csv('ChatGPT_scores.csv')

# Encoding columns
alltrails['city_encoded'], uniques = pd.factorize(alltrails['city_name'])
alltrails['route_type_encoded'], uniques = pd.factorize(alltrails['route_type'])

# Reducing to data used for ML
data = alltrails[["trail_id", "city_encoded", "popularity", "length", "elevation_gain", "route_type_encoded"]]

# Checking for NAs
cols_with_na = data.columns[data.isna().any()].tolist()
print(cols_with_na) # None

# Doing UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

n_clusters = 5  # You can change this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embedding)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=50)
plt.title('UMAP + KMeans Clustering')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()

print(len(labels))
print(len(data))

true_labels = np.array(alltrails["difficulty_rating"])
pred_labels = np.array(labels)

print(np.unique(true_labels))
print(np.unique(pred_labels))

matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()