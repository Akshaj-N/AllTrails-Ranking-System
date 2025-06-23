import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading Data
full = pd.read_csv('../data/full_dataset.csv')
data = full.drop(columns = ["trail_id", "name", "difficulty_rating", "gpt_rating"])

# UMAP
reducer = umap.UMAP(n_neighbors=300, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

# Then clustering using kmeans
n_clusters = 4 # 4 clusters to match other rating systems
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
umap_labels = kmeans.fit_predict(embedding)

alltrails_predictions = np.array(full["difficulty_rating"])
umap_predictions = np.array(umap_labels)
gpt_predictions = np.array(full["gpt_rating"])

# Changing cluster labels to match ranking system
mapping = {0: 2, 1: 4, 2: 3, 3: 1}
transformed_umap = [mapping[x] for x in umap_predictions]


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
    hue='Alltrails_Value',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
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
    hue='GPT_Value',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("GPT Values in UMAP Clustering")
plt.savefig("GPT_Values_in_UMAP.png")
plt.close()


# Plotting UMAP Clusters with Labeled Clusters
sns.scatterplot(
    data=for_plotting,
    x='x', y='y',
    hue='UMAP_Value',         # colors by this variable (continuous)
    # style='cluster',           # shapes by this categorical variable
    palette='viridis',
    s=50
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP Clustering")
plt.savefig("UMAP_clustering.png")
plt.close()


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


# Classification report (includes precision, recall, f1-score)
report = classification_report(df_pred["GPT Prediction"], df_pred["UMAP Prediction"], digits=2)
print(report)


# ---------- SURVEY DATA ----------------

sample_hikes = [10245012, 10265905, 10266148, 10027395, 10006571,
                10033258, 10006208, 10007701, 10289730, 10011593]

sample = df_pred[df_pred["TrailID"].isin(sample_hikes)]

sample.to_csv("sample_output.csv", index=False)

survey = {"AllTrails": [1, 4, 1, 1, 3, 2, 3, 3, 1, 3],
          "FFN": [3, 3, 3, 1, 2, 2, 2, 4, 1, 4],
          "GBDT": [3, 2, 3, 3, 2, 2, 2, 3, 3, 3],
          "UMAP": [3, 2, 3, 2, 2, 2, 2, 2, 1, 2]}

survey_df = pd.DataFrame(survey)
survey_long = survey_df.melt(var_name="Model", value_name="Accuracy (1-5)")

means = survey_long.groupby("Model")["Accuracy (1-5)"].mean()

# Create Scatterplot of Survey Results
plt.figure(figsize=(8, 5))

sns.barplot(data=survey_long, x="Model", y="Accuracy (1-5)", ci="sd", palette="pastel")
sns.stripplot(data=survey_long, x="Model", y="Accuracy (1-5)", jitter=True, size=8, alpha=0.8)

for i, model in enumerate(means.index):
    plt.text(i + 0.1, means[model] + 0.1, f"{means[model]:.2f}", ha='left', va='center', fontsize=10, fontweight='bold')


plt.title("Model Accuracy per Survey Results")
plt.ylim(0.5, 4.5)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.savefig("Model_Accuracy_Survey.png")
plt.close()




