import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading Data
alltrails = pd.read_csv('../alltrails-data.csv')

mapping_dif = {1:1, 3:2, 5:3, 7:4}
alltrails["difficulty_rating"] = [mapping_dif[x] for x in alltrails["difficulty_rating"]]

def convert_to_miles(row):
    if row['units'] == 'i':
        return row['length'] / 1760 # converting to yards
    elif row['units'] == 'm':
        return row['length'] / 1609.344 # converting to meters
    else:
        return None  # or row['length'] if you want to keep unconverted

alltrails['length_miles'] = alltrails.apply(convert_to_miles, axis=1)


chat = pd.read_csv('../chatgpt_new_difficulty_ratings.csv')
print(chat.columns.tolist())

# Encoding columns
alltrails['city_encoded'], uniques = pd.factorize(alltrails['city_name'])
alltrails['route_type_encoded'], uniques = pd.factorize(alltrails['route_type'])

print(alltrails["area_name"].unique())
# Reducing to data used for ML
data = alltrails[["city_encoded", "popularity", "length_miles", "elevation_gain"]]

# Checking for NAs
cols_with_na = data.columns[data.isna().any()].tolist()
print(cols_with_na) # None

# Doing UMAP
reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embedding)

alltrails_labels = np.array(alltrails["difficulty_rating"])
pred_labels = np.array(labels)
chatpred_labels = np.array(chat["difficulty_rating"])

mapping = {0: 1, 1: 4, 2: 2, 3: 3}
transformed = [mapping[x] for x in pred_labels]

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
    "cluster" : transformed,
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
plt.scatter(embedding[:, 0], embedding[:, 1], c=pred_labels, cmap='viridis', s=50)
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
    "TrailID": alltrails["trail_id"],
    "Trail": alltrails["name"],
    "Length": alltrails["length_miles"],
    "elevation_gain": alltrails["elevation_gain"],
    "alltrails_labels": alltrails_labels,
    "Cluster_Class": transformed,
    "Chat Class": chatpred_labels
})

sample_hikes = [10245012, 10265905, 10266148, 10027395, 10006571,
                10033258, 10006208, 10007701, 10289730, 10011593]

sample = df_pred[df_pred["TrailID"].isin(sample_hikes)]

sample.to_csv("output.csv", index=False)




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

survey = {"AllTrails": [1, 4, 1, 1, 3, 2, 3, 3, 1, 3],
          "FFN": [3, 3, 3, 1, 2, 2, 2, 4, 1, 4],
          "Gradient Boosting": [3, 2, 3, 3, 2, 2, 2, 3, 1, 3],
          "UMAP": [3, 2, 3, 2, 2, 2, 2, 2, 1, 2]}

survey_df = pd.DataFrame(survey)
survey_long = survey_df.melt(var_name="Model", value_name="Accuracy (1-5)")

# Create jittered scatterplot
plt.figure(figsize=(8, 5))
sns.barplot(data=survey_long, x="Model", y="Accuracy (1-5)", ci="sd", palette="pastel")
sns.stripplot(data=survey_long, x="Model", y="Accuracy (1-5)", jitter=True, size=8, alpha=0.8)

# Optional: improve formatting
plt.title("Model Accuracy per Survey Results")
plt.ylim(0.5, 4.5)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.show()




