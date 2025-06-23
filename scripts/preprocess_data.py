import pandas as pd
import numpy as np
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, '..', 'data')
alltrails_path = os.path.join(data_dir, "alltrails-data-i.csv")
gpt_path = os.path.join(data_dir, "chatgpt_ratings.csv")
survey_path = os.path.join(data_dir, "survey_hikes.csv")

output_dir = data_dir

# -----------------------------------
# Helper functions
# -----------------------------------
def safe_literal_eval(x):
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
    return df.drop(columns=['features', 'activities'])

def add_engineered_features(df):
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
    return df

# -----------------------------------
# Step 1: Load and merge datasets
# -----------------------------------
df = pd.read_csv(alltrails_path)
gpt_df = pd.read_csv(gpt_path)[['trail_id', 'difficulty_rating']].rename(columns={'difficulty_rating': 'gpt_rating'})
df = df.merge(gpt_df, on='trail_id', how='inner')

# Separate survey hikes
survey_ids = pd.read_csv(survey_path)['trail_id'].tolist()
survey_df = df[df['trail_id'].isin(survey_ids)].copy()
# df = df[~df['trail_id'].isin(survey_ids)].copy()

# Extract sets for one-hot
features_set, activities_set = extract_unique_features_activities(df)
df = add_binary_feature_activity_columns(df, features_set, activities_set)
df = add_engineered_features(df)

# One-hot route_type
df = pd.get_dummies(df, columns=['route_type'], drop_first=True)

# Drop unnecessary columns
drop_cols = ['area_name', 'city_name', 'state_name', 'country_name', '_geoloc', 'visitor_usage', 'difficulty_rating', 'units', 'avg_rating']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])
df.to_csv(os.path.join(output_dir, 'full_dataset.csv'), index=False)

# drop these columns after saving the full_dataset and removing survey hikes
df = df[~df['trail_id'].isin(survey_ids)].copy()
drop_cols_2 = ['trail_id', 'name']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])
# -----------------------------------
# Step 2: Train-test split
# -----------------------------------
y = df['gpt_rating'].values - 1
X = df.drop(columns=['gpt_rating'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Recombine with label
df_train = X_train.copy()
df_train['gpt_rating'] = y_train + 1

df_test = X_test.copy()
df_test['gpt_rating'] = y_test + 1

# -----------------------------------
# Step 3: Handle survey_df too
# -----------------------------------
survey_df = add_binary_feature_activity_columns(survey_df, features_set, activities_set)
survey_df = add_engineered_features(survey_df)
survey_df = pd.get_dummies(survey_df, columns=['route_type'], drop_first=True)
for col in X.columns:
    if col not in survey_df.columns:
        survey_df[col] = 0
survey_df = survey_df[X.columns]

# Save full dataset with survey
# full_data = pd.concat([df_train, df_test, survey_df], axis=0)

# -----------------------------------
# Save all
# -----------------------------------
df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
survey_df.to_csv(os.path.join(output_dir, 'survey_input.csv'), index=False)
# full_data.to_csv(os.path.join(output_dir, 'full_dataset.csv'), index=False)

print("Preprocessing complete. CSVs saved to 'data/' folder.")
