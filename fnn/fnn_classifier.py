import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.confusion_matrix import plot_confusion_matrix
from scripts.survey import predict_on_survey


# Model definition
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
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test.cpu(), preds.cpu()))
    class_names = ['Easy (1)', 'Moderate (2)', 'Hard (3)', 'Very Hard (4)']
    plot_confusion_matrix(y_test.cpu().numpy(), preds.cpu().numpy(), class_names, save_path='fnn_confusion_matrix.png')


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load preprocessed CSVs
    train_path = './data/train.csv'
    test_path = './data/test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df['gpt_rating'].values - 1
    y_test = test_df['gpt_rating'].values - 1

    X_train = train_df.drop(columns=['gpt_rating'])
    X_test = test_df.drop(columns=['gpt_rating'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    model = TrailRatingFNN(input_dim)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    train_model(model, train_loader, criterion, optimizer, epochs=300)
    evaluate_model(model, X_test_tensor, y_test_tensor)

    # Predict and evaluate on survey hikes
    survey_meta_path = './data/survey_full.csv'
    survey_input_path = './data/survey_input.csv'
    survey_results = predict_on_survey(model, survey_input_path, survey_meta_path, scaler)
   
