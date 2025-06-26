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
unsupervised/ - UMAP and clustering code  
README.md - This file  
requirements.txt - Python dependencies

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
   
   **_c. Train the Unsupervised Machine Learning:_**

   python umap/umap_clustering.py

7. **Run predictions on survey trails:**

   python scripts/predict_survey.py

## Contributors

Hannah Wilcox  
Harika Bale  
Akshaj Nevgi
