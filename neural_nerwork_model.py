import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Safe Literal Eval -----------------
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        return []

# ----------------- Feature Set Extraction -----------------
def extract_feature_sets(df):
    df['features'] = df['features'].apply(safe_literal_eval)
    df['activities'] = df['activities'].apply(safe_literal_eval)
    features_set = set(item for sublist in df['features'] for item in sublist)
    activities_set = set(item for sublist in df['activities'] for item in sublist)
    return features_set, activities_set

# ----------------- Preprocessing -----------------
def preprocess(df, drop_cols, target_col, features_set, activities_set, reference_columns=None):
    df = df.dropna(subset=[target_col])
    df['features'] = df['features'].apply(safe_literal_eval)
    df['activities'] = df['activities'].apply(safe_literal_eval)

    for feature in features_set:
        df[f'feature_{feature}'] = df['features'].apply(lambda x: int(feature in x))
    for activity in activities_set:
        df[f'activity_{activity}'] = df['activities'].apply(lambda x: int(activity in x))

    df = df.drop(columns=['features', 'activities'])
    df = pd.get_dummies(df, columns=['route_type', 'visitor_usage', 'units'], drop_first=True)

    df['steepness'] = df['elevation_gain'] / (df['length'] + 1e-5)
    df['gain_x_length'] = df['elevation_gain'] * df['length']
    df['gain_x_steepness'] = df['elevation_gain'] * df['steepness']
    df['length_x_steepness'] = df['length'] * df['steepness']
    df['elevation_gain_sq'] = df['elevation_gain'] ** 2
    df['length_sq'] = df['length'] ** 2
    df['steepness_sq'] = df['steepness'] ** 2
    df['log_elevation_gain'] = np.log1p(df['elevation_gain'])
    df['log_length'] = np.log1p(df['length'])
    df['sqrt_elevation_gain'] = np.sqrt(df['elevation_gain'])
    df['sqrt_length'] = np.sqrt(df['length'])
    df['steepness_bin'] = pd.cut(df['steepness'], bins=[-np.inf, 10, 30, 100, np.inf], labels=[0, 1, 2, 3]).astype(int)

    df['rating_class'] = df[target_col].round().astype(int) - 1
    y = df['rating_class'].values
    X = df.drop(columns=drop_cols + [target_col])
    X = X.select_dtypes(include=[np.number])

    if reference_columns is not None:
        for col in reference_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[reference_columns]

    return X, y

# ----------------- Load Data -----------------
def load_data(alltrails_path, gpt_path):
    df = pd.read_csv(alltrails_path)
    gpt_df = pd.read_csv(gpt_path)[['trail_id', 'estimated_strenuousness_score']]
    gpt_df = gpt_df.rename(columns={'estimated_strenuousness_score': 'gpt_rating'})
    df = df.merge(gpt_df, on='trail_id', how='inner')
    return df

# ----------------- Neural Network -----------------
class TrailRatingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TrailRatingClassifier, self).__init__()
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

# ----------------- Training -----------------
def train_model(model, train_loader, val_tensor, val_targets, class_weights, epochs=300, lr=0.0005, patience=25):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_targets)
            val_losses.append(val_loss.item())

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# ----------------- Data distribution --------------------------
def plot_class_distribution(y, title):
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()



# ----------------- Main -----------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    alltrails_path = 'alltrails-data.csv'
    gpt_path = 'estimated_strenuousness_scores.csv'
    drop_cols = ['name', 'trail_id', 'area_name', 'city_name', 'state_name', 'country_name', '_geoloc',
                 'avg_rating', 'difficulty_rating']
    target_col = 'gpt_rating'

    df = load_data(alltrails_path, gpt_path)
    df = df.dropna(subset=[target_col])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    features_set, activities_set = extract_feature_sets(train_df)

    X_train, y_train = preprocess(train_df, drop_cols, target_col, features_set, activities_set)
    X_val, y_val = preprocess(val_df, drop_cols, target_col, features_set, activities_set, reference_columns=X_train.columns)
    X_test, y_test = preprocess(test_df, drop_cols, target_col, features_set, activities_set, reference_columns=X_train.columns)

    # After splitting your data:
    print("Train class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    plot_class_distribution(y_train, "Training Set Class Distribution")

    print("Validation class distribution:")
    unique, counts = np.unique(y_val, return_counts=True)
    print(dict(zip(unique, counts)))
    plot_class_distribution(y_val, "Validation Set Class Distribution")

    print("Test class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique+1, counts)))
    plot_class_distribution(y_test, "Test Set Class Distribution")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train)), batch_size=128, shuffle=True)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test)

    model = TrailRatingClassifier(input_dim=X_train.shape[1], num_classes=5)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    train_losses, val_losses = train_model(model, train_loader, X_val_tensor, y_val_tensor, class_weights_tensor)

    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_test_tensor), dim=1).numpy()

    exact_acc = np.mean(test_preds == y_test)
    within_1_acc = np.mean(np.abs(test_preds - y_test) <= 1)
    print(f"\nTest Set Evaluation:\nExact Match Accuracy: {exact_acc*100:.2f}%\nWithin ±1 Accuracy: {within_1_acc*100:.2f}%")

    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nClassification Report:\n", classification_report(y_test, test_preds, target_names=["1","2","3","4","5"]))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.7)
    plt.plot(val_losses, label="Val Loss", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

if __name__ == "__main__":
    main()

