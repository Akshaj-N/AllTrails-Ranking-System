import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import ast
import random
import matplotlib.pyplot as plt
import seaborn as sns


# load data
def load_data(alltrails_path, gpt_path):
    """
    Loads and merges AllTrails and ChatGPT rating data.

    Parameters:
        alltrails_path (str): Path to alltrails-data.csv
        gpt_path (str): Path to chatgpt_ratings.csv

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    df = pd.read_csv(alltrails_path)
    gpt_df = pd.read_csv(gpt_path)[['trail_id', 'difficulty_rating']]
    gpt_df = gpt_df.rename(columns={'difficulty_rating': 'gpt_rating'})
    df = df.merge(gpt_df, on='trail_id', how='inner')
    return df

# Remove survey hikes
def separate_survey_hikes(df, survey_path):
    """
    Removes survey hikes from main DataFrame and returns both DataFrames.

    Parameters:
        df (pd.DataFrame): Full dataset
        survey_path (str): Path to Survey_hikes.csv

    Returns:
        pd.DataFrame: Cleaned df (without survey hikes)
        pd.DataFrame: survey_df (only survey hikes)
    """
    survey_hikes = pd.read_csv(survey_path)
    survey_ids = set(survey_hikes['trail_id'].tolist())

    survey_df = df[df['trail_id'].isin(survey_ids)].copy()
    df = df[~df['trail_id'].isin(survey_ids)].copy()

    print(f"\nRemoved {len(survey_df)} survey hikes from training data.")
    print(f"Remaining dataset size: {len(df)}")

    return df, survey_df


# Print deatils about data
def desc_data(df):
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of Columns: {len(df.columns.tolist())}")

# Some Analysis
def check_missing(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())

def sample_rows(df, n=5):
    print(f"\nSample {n} rows:")
    print(df.head(n))

def target_distribution(df, target_col='gpt_rating'):
    print(f"\nDistribution of {target_col}:")
    print(df[target_col].value_counts().sort_index())


# Preprocessing
def safe_literal_eval(x):
    # Convert string representation of list to actual list
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        return []

def extract_unique_features_activities(df):
    df['features'] = df['features'].apply(safe_literal_eval)
    df['activities'] = df['activities'].apply(safe_literal_eval)
    features_set = set(item for sublist in df['features'] for item in sublist)
    activities_set = set(item for sublist in df['activities'] for item in sublist)
    
    return features_set, activities_set

def add_binary_feature_activity_columns(df, features_set, activities_set):
    df['features'] = df['features'].apply(safe_literal_eval)
    df['activities'] = df['activities'].apply(safe_literal_eval)
    for feature in features_set:
        df[f'feature_{feature}'] = df['features'].apply(lambda x: int(feature in x))
    for activity in activities_set:
        df[f'activity_{activity}'] = df['activities'].apply(lambda x: int(activity in x))
    df = df.drop(columns=['features', 'activities'])
    return df

def add_engineered_features(df):
    # Basic derived feature
    df['steepness'] = df['elevation_gain'] / (df['length'] + 1e-5)

    # Interaction terms
    df['gain_x_length'] = df['elevation_gain'] * df['length']
    df['gain_x_steepness'] = df['elevation_gain'] * df['steepness']
    df['length_x_steepness'] = df['length'] * df['steepness']

    # Polynomial features
    df['elevation_gain_sq'] = df['elevation_gain'] ** 2
    df['length_sq'] = df['length'] ** 2
    df['steepness_sq'] = df['steepness'] ** 2

    # Log transforms
    df['log_elevation_gain'] = np.log1p(df['elevation_gain'])
    df['log_length'] = np.log1p(df['length'])

    # Square root transforms
    df['sqrt_elevation_gain'] = np.sqrt(df['elevation_gain'])
    df['sqrt_length'] = np.sqrt(df['length'])

    # Binned steepness (custom bins)
    df['steepness_bin'] = pd.cut(
        df['steepness'],
        bins=[-np.inf, 10, 30, 100, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df


class TrailRatingFNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(TrailRatingFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    
def train_model(model, train_loader, criterion, optimizer, epochs=300):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    # classification report:
    from sklearn.metrics import classification_report
    print(classification_report(y_test.cpu(), preds.cpu()))

    # confusion matrix plotting
    class_names = ['Easy (1)', 'Moderate (2)', 'Hard (3)', 'Very Hard (4)']
    plot_confusion_matrix(y_test.cpu().numpy(), preds.cpu().numpy(), class_names)


def predict_on_survey(model, survey_df, features_set, activities_set, scaler, device='cpu'):
    # Drop unwanted columns same as training
    drop_cols = [
        'trail_id', 'area_name', 'city_name', 'state_name', 'country_name',
        '_geoloc', 'visitor_usage', 'difficulty_rating', 'units', 'avg_rating'
    ]
    survey_df_proc = survey_df.drop(columns=drop_cols, errors='ignore').copy()

    # One-hot encode features/activities
    survey_df_proc = add_binary_feature_activity_columns(survey_df_proc, features_set, activities_set)
    survey_df_proc = add_engineered_features(survey_df_proc)

    # One-hot encode 'route_type'
    survey_df_proc = pd.get_dummies(survey_df_proc, columns=['route_type'], drop_first=True)

    # Align columns with training features (add missing cols with 0)
    for col in X.columns:
        if col not in survey_df_proc.columns:
            survey_df_proc[col] = 0
    survey_df_proc = survey_df_proc[X.columns]  # reorder columns to match X

    # Scale features
    X_survey_scaled = scaler.transform(survey_df_proc)

    # Convert to tensor
    X_survey_tensor = torch.tensor(X_survey_scaled, dtype=torch.float32).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(X_survey_tensor)
        _, preds = torch.max(outputs, 1)  # get predicted class indices (0..3)

    # Map classes back to ratings [1,3,5,7]
    predicted_rating_map = {0: 1, 1: 2, 2: 3, 3: 4}

    # Map classes back to ratings [1,3,5,7]
    difficulty_rating_map = {1: 1, 3: 2, 5: 3, 7: 4}

    survey_df['predicted_rating'] = preds.cpu().numpy()
    survey_df['predicted_rating'] = survey_df['predicted_rating'].map(predicted_rating_map)
    survey_df['difficulty_rating'] = survey_df['difficulty_rating'].map(difficulty_rating_map)

    # Match between difficulty_rating and predicted_rating_mapped (both on [1,3,5,7])
    survey_df['Match'] = survey_df['difficulty_rating'] == survey_df['predicted_rating']

    survey_results_filename = './fnn/fnn_survey_predictions.csv'
    
    # Select and save columns
    survey_df[['trail_id', 'name', 'difficulty_rating', 'gpt_rating', 'predicted_rating', 'Match']].to_csv(survey_results_filename, index=False)
    print(f"Survey results are saved to file: {survey_results_filename}")

    return survey_df


def plot_rating_distributions(
    gpt_counts, gpt_labels,
    alltrails_counts, alltrails_labels,
    save_path='./data/rating_distribution_comparison.png'
):
    gpt_color = '#A3C4A8'       
    alltrails_color = '#345C4F' 
    bg_color = 'white'       
    text_color = '#2E2E2E' 

    x = np.arange(len(gpt_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bars1 = ax.bar(x - width/2, gpt_counts, width, label='GPT Rating', color=gpt_color)
    bars2 = ax.bar(x + width/2, alltrails_counts, width, label='AllTrails Rating', color=alltrails_color)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {label}' for label in gpt_labels], color=text_color)
    ax.set_ylabel('Number of Trails', color=text_color)
    ax.set_title('Class Distribution: GPT vs AllTrails', color=text_color)
    ax.legend(facecolor=bg_color, edgecolor='white', labelcolor=text_color)

    # Add counts on top
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, color=text_color)

    plt.tight_layout()
    plt.savefig(save_path, facecolor=bg_color)
    plt.close()
    print(f"Saved AllTrails-themed plot to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='./fnn/fnn_confusion_matrix.png'):
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
    print(f" Confusion matrix saved as: {save_path}")



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    alltrails_path = 'data/alltrails-data-i.csv'
    gpt_path = 'data/chatgpt_ratings.csv'
    survey_path = 'data/survey_full.csv'
    df = load_data(alltrails_path, gpt_path)
    df, survey_df = separate_survey_hikes(df, survey_path)

    desc_data(df)
    # check_missing(df)
    # sample_rows(df)
    target_distribution(df)

    gpt_counts = [359, 853, 851, 1191]
    gpt_labels = ['1', '2', '3', '4']

    alltrails_counts = [869, 1433, 771, 181]
    alltrails_labels = ['1', '3', '5', '7']  

    plot_rating_distributions(gpt_counts, gpt_labels, alltrails_counts, alltrails_labels)

    drop_cols = [
    'trail_id', 'name', 'area_name', 'city_name', 'state_name', 'country_name',
    '_geoloc', 'visitor_usage', 'difficulty_rating', 'units', 'avg_rating'
    ]

    df = df.drop(columns=drop_cols)
    print("\nAfter dropping unwanted columns:")
    print(df.columns.tolist())

    # Extract unique features/activities and encode as binary columns
    features_set, activities_set = extract_unique_features_activities(df)
    df = add_binary_feature_activity_columns(df, features_set, activities_set)
    df = add_engineered_features(df)
    
    # Check shape and columns after encoding
    print("\nAfter one-hot encoding features and activities:")
    desc_data(df)

    categorical_cols = ['route_type']

    # Step 1: Separate X and y
    y = df['gpt_rating'].values - 1
    X = pd.get_dummies(df.drop(columns=['gpt_rating']), columns=categorical_cols, drop_first=True)

    # Step 2: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: Standardize (scale) features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print to verify shapes
    # print(f"\nTraining samples: {X_train.shape[0]}")
    # print(f"Test samples: {X_test.shape[0]}")

    input_dim = X_train_scaled.shape[1]
    model = TrailRatingFNN(input_dim)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    train_model(model, train_loader, criterion, optimizer, epochs=300)
    evaluate_model(model, X_test_tensor, y_test_tensor)

    # Predict and evaluate on survey hikes
    survey_results = predict_on_survey(model, survey_df, features_set, activities_set, scaler)




