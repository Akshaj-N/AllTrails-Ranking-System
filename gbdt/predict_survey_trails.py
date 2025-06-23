import numpy as np
import pandas as pd
import joblib

# loading the saved models and objects
models = joblib.load('gbdt_models.pkl')
log_class_priors = joblib.load('log_class_priors.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# parameters
n_classes = len(models)
learning_rate = 0.05
features = ['popularity', 'length', 'elevation_gain', 'route_type', 'visitor_usage', 'num_reviews']

# loading the data
data = pd.read_csv('alltrails-data.csv')
chatgpt_ratings = pd.read_csv('ChatGPT_scores.csv')
data = data.merge(chatgpt_ratings[['trail_id', 'difficulty_rating']], on='trail_id', suffixes=('', '_label'))
data = data.dropna(subset=['difficulty_rating'])

# survey trails
trail_ids = [10245012, 10265905, 10266148, 10027395, 10006571, 10033258, 10006208, 10007701, 10289730, 10011593]

survey_trails = data[data['trail_id'].isin(trail_ids)].copy()
survey_trails = survey_trails.dropna(subset=['difficulty_rating'])
survey_trails['route_type'] = label_encoder.transform(survey_trails['route_type'].astype(str))

# building feature matrix
X_selected = survey_trails[features].copy()
X_selected['route_type'] = survey_trails['route_type']

F_selected = np.tile(log_class_priors, (X_selected.shape[0], 1))
for k in range(n_classes):
    for model in models[k]:
        F_selected[:, k] += learning_rate * model.predict(X_selected.values)
predicted_classes = np.argmax(F_selected, axis=1)

# mapping difficulty ratings according to user scores
difficulty_remap = {1: 1, 3: 2, 5: 3, 7: 4}
survey_trails['difficulty_rating'] = survey_trails['difficulty_rating'].map(difficulty_remap)
actual_classes = survey_trails['difficulty_rating'].astype(int) - 1

match = predicted_classes == actual_classes
survey_trails['Actual Difficulty'] = actual_classes + 1
survey_trails['Predicted Difficulty'] = predicted_classes + 1
survey_trails['Match'] = match

output_df = survey_trails[['trail_id', 'name', 'Actual Difficulty', 'Predicted Difficulty', 'Match']]
print(output_df.to_string(index = False))

output_df.to_csv('survey_trail_predictions.csv', index = False)
