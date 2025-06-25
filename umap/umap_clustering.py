import pandas as pd
import umap
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data
full = pd.read_csv('../data/full_dataset.csv')
data = full.drop(columns = ["trail_id", "name", "difficulty_rating", "gpt_rating"])

# Fix labels of difficulty rating to match 1-4 system
dif_mapping = {1:1, 3:2, 5:3, 7:4}
full["difficulty_rating"] = [dif_mapping[x] for x in full["difficulty_rating"]]

# UMAP for reducing dimensionality
reducer = umap.UMAP(n_neighbors=300, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

#  Kmeans for clustering, ChatGPT helped with debugging
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)

    # Initialize centroids
    n_samples, n_features = X.shape
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[initial_indices]

    for iteration in range(max_iters):
        # Assign each point to centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Make new centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return labels

n_clusters = 4 # 4 clusters to match other rating systems
umap_labels = kmeans(embedding, k=n_clusters)

# Summarizing all rating systems
alltrails_predictions = np.array(full["difficulty_rating"])
umap_predictions = np.array(umap_labels)
gpt_predictions = np.array(full["gpt_rating"])

# Changing cluster labels to match ranking system as numbers were randomly placed
mapping = {0: 2, 1: 3, 2: 1, 3: 4}
transformed_umap = [mapping[x] for x in umap_predictions]

# ---------- Plotting ---------------------------------

# Merging labels for plotting
for_plotting = pd.DataFrame({
    "x" : embedding[:, 0],
    "y" : embedding[:, 1],
    "UMAP_Value" : transformed_umap,
    "Alltrails_Value": alltrails_predictions,
    "GPT_Value": gpt_predictions
})

# Plotting AllTrails over UMAP
sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='Alltrails_Value',
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("AllTrails Values in UMAP Clustering")
plt.savefig("Alltrails_in_UMAP.png")
plt.close()

# Plotting GPT values over UMAP
sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='GPT_Value',
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("GPT Values in UMAP Clustering")
plt.savefig("GPT_Values_in_UMAP.png")
plt.close()

# Plotting GPT values over UMAP
sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='GPT_Value',
    style = 'Alltrails_Value',
    palette='viridis',
    s=50
)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("GPT and AllTrails Values in UMAP Clustering")
plt.savefig("Both_Values_in_UMAP.png", bbox_inches='tight')
plt.close()


# Plotting UMAP Clusters with Labeled Clusters

sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='UMAP_Value',
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP Clustering")
plt.savefig("UMAP_clustering.png")
plt.close()

# -------------------- Evaluating Model --------------------------------------


df_pred = pd.DataFrame({
    "TrailID": full["trail_id"],
    "Trail": full["name"],
    "Length": full["length"],
    "elevation_gain": full["elevation_gain"],
    "AllTrails Prediction": alltrails_predictions,
    "UMAP Prediction": transformed_umap,
    "GPT Prediction": gpt_predictions
})

# Calculating Accuracy
accuracy_gpt = (df_pred["UMAP Prediction"] == df_pred["GPT Prediction"]).mean()
print(f"Accuracy of UMAP matching GPT: {accuracy_gpt:.2%}")

accuracy_AT = (df_pred["UMAP Prediction"] == df_pred["AllTrails Prediction"]).mean()
print(f"Accuracy of UMAP matching AllTrails: {accuracy_AT:.2%}")

# Classification report
report = classification_report(df_pred["GPT Prediction"], df_pred["UMAP Prediction"], digits=2)
print(report)

conf_mat = confusion_matrix(df_pred["GPT Prediction"], df_pred["UMAP Prediction"])
print(conf_mat)


# ---------- SURVEY DATA ----------------

sample_hikes = [10245012, 10265905, 10266148, 10027395, 10006571,
                10033258, 10006208, 10007701, 10289730, 10011593]

sample = df_pred[df_pred["TrailID"].isin(sample_hikes)]

sample.to_csv("sample_output.csv", index=False)