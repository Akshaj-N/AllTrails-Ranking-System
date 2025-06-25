import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

survey = {"AllTrails": [1, 4, 1, 1, 3, 2, 3, 3, 1, 3],
          "FNN": [3, 3, 3, 1, 2, 2, 2, 4, 1, 4],
          "GBDT": [3, 2, 3, 3, 2, 2, 2, 3, 3, 3],
          "UMAP": [3, 2, 3, 2, 2, 2, 2, 2, 1, 2]}

survey_df = pd.DataFrame(survey)
survey_long = survey_df.melt(var_name="Model", value_name="Accuracy (1-5)")

means = survey_long.groupby("Model")["Accuracy (1-5)"].mean()

# Create Scatterplot of Survey Results
plt.figure(figsize=(8, 5))

sns.barplot(data=survey_long, x="Model", y="Accuracy (1-5)", errorbar='sd', palette="pastel")
sns.stripplot(data=survey_long, x="Model", y="Accuracy (1-5)", jitter=True, size=8, alpha=0.8)

for i, model in enumerate(means.index):
    plt.text(i + 0.1, means[model] + 0.1, f"{means[model]:.2f}", ha='left', va='center', fontsize=10, fontweight='bold')

plt.title("Model Accuracy per Survey Results")
plt.ylim(0.5, 4.5)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.savefig("Model_Accuracy_Survey.png")
plt.close()
