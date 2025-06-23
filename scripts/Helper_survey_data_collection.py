import pandas as pd

def merge_survey_with_alltrails(alltrails_path, survey_path, gpt_path, output_path):
    # Load full AllTrails dataset
    alltrails_df = pd.read_csv(alltrails_path)
    print(f"✅ AllTrails dataset loaded: {alltrails_df.shape[0]} rows, {alltrails_df.shape[1]} columns")

    # Load survey hikes (subset of trail_ids)
    survey_df = pd.read_csv(survey_path)
    print(f"✅ Survey hikes loaded: {survey_df.shape[0]} rows, {survey_df.shape[1]} columns")

    # Filter for survey hikes only
    survey_ids = survey_df['trail_id'].unique()
    survey_full_df = alltrails_df[alltrails_df['trail_id'].isin(survey_ids)].copy()
    print(f"✅ Found {survey_full_df.shape[0]} matching rows in AllTrails data")

    # Load GPT difficulty ratings
    gpt_df = pd.read_csv(gpt_path)[['trail_id', 'difficulty_rating']]
    gpt_df = gpt_df.rename(columns={'difficulty_rating': 'gpt_rating'})

    # Merge GPT difficulty ratings into survey_full_df
    survey_full_df = survey_full_df.merge(gpt_df, on='trail_id', how='left')
    print(f"✅ After merging GPT ratings: {survey_full_df.shape[0]} rows")

    # Save result
    survey_full_df.to_csv(output_path, index=False)
    print(f"📁 Saved merged survey hike data to: {output_path}")

if __name__ == "__main__":
    alltrails_path = './data/alltrails-data-i.csv'
    survey_path = './data/Survey_hikes.csv'
    gpt_path = './data/chatgpt_ratings.csv'
    output_path = './data/survey_full.csv'

    merge_survey_with_alltrails(alltrails_path, survey_path, gpt_path, output_path)
