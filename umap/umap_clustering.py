import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading Data
alltrails = pd.read_csv('../alltrails-data.csv')
chat = pd.read_csv('../chatgpt_new_difficulty_ratings.csv')
print(chat.columns.tolist())

# Encoding columns
alltrails['city_encoded'], uniques = pd.factorize(alltrails['city_name'])
alltrails['route_type_encoded'], uniques = pd.factorize(alltrails['route_type'])

print(alltrails["area_name"].unique())
# Reducing to data used for ML
data = alltrails[["city_encoded", "popularity", "length", "elevation_gain"]]

# Checking for NAs
cols_with_na = data.columns[data.isna().any()].tolist()
print(cols_with_na) # None

# Doing UMAP
reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embedding)

alltrails_labels = np.array(alltrails["difficulty_rating"])
pred_labels = np.array(labels)
chatpred_labels = np.array(chat["difficulty_rating"])

unique, counts = np.unique(alltrails_labels, return_counts=True)

# Create a bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=unique, y=counts, palette='pastel')
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Values in AllTrails Data")
plt.xticks(ticks=range(len(unique)), labels=[f"Class {u}" for u in unique])
plt.tight_layout()
plt.show()


unique, counts = np.unique(chatpred_labels, return_counts=True)

# Create a bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=unique, y=counts, palette='pastel')
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Values in ChatGPT Data")
plt.xticks(ticks=range(len(unique)), labels=[f"Class {u}" for u in unique])
plt.tight_layout()
plt.show()


for_plotting = pd.DataFrame({
    "x" : embedding[:, 0],
    "y" : embedding[:, 1],
    "cluster" : pred_labels,
    "Alltrails_Value": alltrails_labels,
    "ChatGPT_Value": chatpred_labels
})

sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='Alltrails_Value',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("AllTrails Values in UMAP Clustering")
plt.show()

sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='ChatGPT_Value',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("Chat Values in UMAP Clustering")
plt.show()


sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='cluster',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP Clustering")
plt.show()


# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=50)
plt.title('UMAP + KMeans Clustering')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.savefig("clustering.png")
plt.close()

print(len(labels))
print(len(data))
print(len(chat))

print(f"Unique Values in Alltrails:", np.unique(alltrails_labels))
print(f"Unique Values in Umap predictions:", np.unique(pred_labels))
print(f"Unique Values in ChatGPT predictions:", np.unique(chatpred_labels))

unique, counts = np.unique(pred_labels, return_counts=True)

# Create a bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=unique, y=counts, palette='pastel')
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Values in Data")
plt.xticks(ticks=range(len(unique)), labels=[f"Class {u}" for u in unique])
plt.tight_layout()
plt.show()

df_pred = pd.DataFrame({
    "Trail": alltrails["name"],
    "Length": alltrails["length"],
    "elevation_gain": alltrails["elevation_gain"],
    "alltrails_labels": alltrails_labels,
    "Predicted_Class": pred_labels
})

sampled_df = df_pred.groupby('Predicted_Class').sample(n=4, random_state=42)

sampled_df.to_csv("output.csv", index=False)




alltrails_matrix = confusion_matrix(alltrails_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(alltrails_matrix, annot=True, fmt="d", cmap="Greens",
            xticklabels=np.unique(pred_labels), yticklabels=np.unique(alltrails_labels))
plt.title("Alltrails Confusion Matrix")
plt.xlabel("Umap Predicted Label")
plt.ylabel("Alltrails Predicted Label")
plt.tight_layout()
plt.savefig("alltrails_confusion_matrix.png")
plt.close()

matrix = confusion_matrix(chatpred_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Chat Confusion Matrix")
plt.xlabel("Umap Predicted Label")
plt.ylabel("ChatGPT Predicted Label")
plt.tight_layout()
plt.savefig("chat_confusion_matrix.png")
plt.close()

matrix = confusion_matrix(alltrails_labels, chatpred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Chat Confusion Matrix")
plt.xlabel("Chat Predicted Label")
plt.ylabel("Alltrails Predicted Label")
plt.tight_layout()
plt.savefig("chat_v_alltrails_confusion_matrix.png")
plt.close()