# Trail Difficulty Estimation Using AI

This project predicts the difficulty of hiking trails using structured trail metadata and user-generated content. We explore three modeling approaches -- Feedforward Neural Networks (FNN), Gradient Boosting Decision Trees (GBDT), and Unsupervised Learning to classify trails into four difficulty levels. The project uses data from AllTrails and custom-labeled datasets (ChatGPT).

## Motivation

AllTrails' difficulty ratings are often vague or inconsistent, underestimating the challenge of certain hikes. Our project aims to provide a more accurate and data-driven classification of trail difficulty by leveraging metadata, user experiences, and machine learning techniques.

## Data Sources

- [AllTrails Dataset](https://github.com/j-ane/trail-data/blob/master/alltrails-data.csv) – Metadata on 3000+ hiking trails.
- ChatGPT Ratings – Generated difficulty labels using AI based on user reviews/comments from the web.
- Survey Ratings – Collected from 10 AllTrails users familiar with specific trails.

## Modeling Approaches

- **FNN (Feedforward Neural Network):** Supervised deep learning model trained on engineered features with class-weighted loss to handle imbalance.
- **GBDT (Gradient Boosting Decision Trees):** Interpretable, ensemble-based model.
- **Unsupervised Learning:** UMAP and clustering used to explore patterns in trail characteristics.

## Repository Structure

data/ - Contains CSV files (raw and preprocessed)  
scripts/ - Python scripts for preprocessing, training, and evaluation  
fnn/ - Code and models for Feedforward Neural Network  
gbdt/ - Code for Gradient Boosting models  
umap/ - UMAP and clustering code  
README.md - This file  
requirements.txt - Python dependencies

```plaintext
CS5100_Project/
├── __pycache__/                      # Python bytecode cache
├── .idea/                           # IDE config files (for PyCharm/IntelliJ)
├── data/
│   ├── alltrails-data-i.csv         # Raw hiking trail metadata from AllTrails
│   ├── chatgpt_ratings.csv          # GPT-generated trail difficulty ratings
│   ├── full_dataset.csv             # Fully preprocessed dataset used by umap
│   ├── rating_distribution_comparison.png  # Chart comparing original vs GPT ratings
│   ├── survey_full.csv              # Combined dataset of survey trails with metadata and ratings
│   ├── survey_results.csv           # Final results comparing model predictions vs survey responses
├── fnn/
│   ├── fnn_classifier.py            # Main script to train and evaluate the FNN model
│   ├── fnn_confusion_matrix.png     # Confusion matrix for FNN model predictions
│   ├── fnn_survey_predictions.csv   # FNN model predictions on survey trails
├── gbdt/
│   ├── confusion_matrix_gdbt.png    # Confusion matrix for GBDT predictions
│   ├── gbdt_models.pkl              # Saved GBDT model files
│   ├── gdbt.py                      # Script to train the Gradient Boosting model
│   ├── label_encoder.pkl            # Encoded class labels used in GBDT model
│   ├── log_class_priors.pkl         # Prior class probability logs for boosting initialization
│   ├── model_predictions.csv        # GBDT predictions on test set
│   ├── predict_survey_trails.py     # Predict survey trail difficulty using trained GBDT model
│   ├── survey_trail_predictions.csv # GBDT predictions on survey trails
├── scripts/
│   ├── __init__.py                  # Makes scripts directory a Python package
│   ├── Model_Accuracy_survey.png    # Comparison of model accuracies on survey trails
│   ├── preprocess_data.py           # Script to load, merge, clean, and engineer features
│   ├── survey_results.py            # Aggregates model predictions and compares with survey results
├── umap/
│   ├── Alltrails_in_UMAP.png        # UMAP visualization for all trails
│   ├── Both_Values_in_UMAP.png      # UMAP with both survey and GPT values
│   ├── GPT_Values_in_UMAP.png       # UMAP visualization using only GPT ratings
│   ├── sample_output.csv            # Output file with clustered trail samples
│   ├── UMAP_clustering.png          # Cluster visualization from UMAP
│   ├── umap_clustering.py           # Script to perform unsupervised clustering on trail data
├── .gitignore                       # Files/directories to be ignored by Git
├── README.md                        # Project overview and setup guide
├── requirements.txt                 # List of required Python packages
```

## Installation & Setup

Follow these steps to set up and run the project locally.

1. **Clone the repository:**

   git clone https://github.com/hannahw101101/CS5100_Project.git

   cd CS5100_Project

2. **Create and activate a virtual environment (optional but recommended):**
   python3 -m venv venv

   #### On Mac use:

   source venv/bin/activate

   #### On Windows use:

   venv\Scripts\activate

3. **Install dependencies:**

   pip install -r requirements.txt

4. **Run data preprocessing to prepare training, testing, and survey datasets:**

   python scripts/preprocess_data.py

5. **Train the models:**

   **_a. Train the Feedforward Neural Network (FNN):_**

   python fnn/fnn_classifier.py

   **_b.(i) Train the Gradient Boosting Decision Tree:_**

   python gbdt/gdbt.py

   **_b.(ii) Predict the trails using GBDT:_**

   python gbdt/predict_survey_trails.py

   **_c.(iii) Train the Unsupervised Machine Learning:_**

   python umap/umap_clustering.py

6. **Process survey results:**

   python scripts/survey_results.py

## Contributors

Hannah Wilcox
Harika Bale
Akshaj Nevgi
```
