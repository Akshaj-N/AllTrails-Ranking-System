import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.special import softmax
import joblib

# loading the dataset
data = pd.read_csv('alltrails-data.csv')
chatgpt_ratings = pd.read_csv('ChatGPT_scores.csv')

data = data.merge(chatgpt_ratings[['trail_id', 'difficulty_rating']], on='trail_id', suffixes=('', '_label'))
data = data.dropna(subset=['difficulty_rating_label'])

# feature selection
features = ['popularity', 'length', 'elevation_gain', 'route_type', 'visitor_usage', 'num_reviews']

X = data.loc[:, features].copy()
y = data['difficulty_rating_label'].astype(int) - 1

# encoding route_type as integers
label_encoder = LabelEncoder()
X['route_type'] = label_encoder.fit_transform(X['route_type'].astype(str)) 

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.2, random_state = 42, stratify = y)

# parameters
n_classes = len(np.unique(y))
n_estimators = 40
learning_rate = 0.05

# one hot encoding
y_train_onehot = np.zeros((y_train.size, n_classes))
for i in range(y_train.size):
    class_idx = y_train[i]
    y_train_onehot[i, class_idx] = 1

# initialization of logit using class priors
class_counts = np.bincount(y_train, minlength=n_classes) + 1e-8  # this is to avoid log(0)
class_priors = class_counts / np.sum(class_counts)
log_class_priors = np.log(class_priors)

# initializing logits F with log_class_priors (base score)
F = np.tile(log_class_priors, (X_train.shape[0], 1))  # shape: (n_samples, n_classes)

# storing all the estimators per class
models = []
for _ in range(n_classes):
    models.append([])

# Gradient Boosting Loop
for m in range(n_estimators):
    probs = softmax(F, axis=1)
    gradients = y_train_onehot - probs  # gradient of softmax cross-entropy

    for k in range(n_classes):
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X_train, gradients[:, k])
        pred = model.predict(X_train)
        F[:, k] += learning_rate * pred
        models[k].append(model)

# predicting on the test data
F_test = np.tile(log_class_priors, (X_test.shape[0], 1))

# adding predictions from each boosting round
for k in range(n_classes):
    for model in models[k]:
        F_test[:, k] += learning_rate * model.predict(X_test)

# final predicted class is the argmax of logits
y_pred = np.argmax(F_test, axis=1)

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# saving y_test and y_pred to CSV
output = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred})
output.to_csv('model_predictions.csv', index=False)

# saving the models and the metadata
joblib.dump(models, 'gbdt_models.pkl')
joblib.dump(log_class_priors, 'log_class_priors.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
