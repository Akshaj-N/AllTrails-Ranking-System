import pandas as pd

# File paths
alltrails_raw_path = '../data/alltrails-data-i.csv'
train_path = '../data/train.csv'
test_path = '../data/test.csv'
survey_path = '../data/survey_input.csv'
full_path = '../data/full_dataset.csv'

# Load datasets
alltrails_df = pd.read_csv(alltrails_raw_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
survey_df = pd.read_csv(survey_path)
full_df = pd.read_csv(full_path)

# Print sizes
print("📂 Dataset Summary:")
print(f"Original AllTrails raw data: {alltrails_df.shape[0]} rows, {alltrails_df.shape[1]} columns")
print(f"Train set:                 {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test set:                  {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"Survey set:                {survey_df.shape[0]} rows, {survey_df.shape[1]} columns")
print(f"Full dataset (with survey):{full_df.shape[0]} rows, {full_df.shape[1]} columns")
