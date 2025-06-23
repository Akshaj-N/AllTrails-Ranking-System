import pandas as pd
import torch

def predict_on_survey(model, survey_input_path, survey_meta_path, scaler, device='cpu'):
    """
    Predicts trail difficulty on preprocessed survey data and merges with metadata.

    Parameters:
        model (torch.nn.Module): Trained model
        survey_input_path (str): Path to preprocessed feature CSV (survey_input.csv)
        survey_meta_path (str): Path to original metadata CSV (survey_full.csv)
        scaler (sklearn Scaler): StandardScaler used on training data
        device (str): Device to run inference ('cpu' or 'cuda')

    Returns:
        pd.DataFrame: Merged DataFrame with predictions and metadata
    """
    # Load preprocessed input features
    survey_input = pd.read_csv(survey_input_path)
    X_survey_scaled = scaler.transform(survey_input)

    X_survey_tensor = torch.tensor(X_survey_scaled, dtype=torch.float32).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(X_survey_tensor)
        _, preds = torch.max(outputs, 1)

    # Map predicted class to rating
    predicted_rating_map = {0: 1, 1: 2, 2: 3, 3: 4}
        
    # Map classes back to ratings [1,3,5,7]
    difficulty_rating_map = {1: 1, 3: 2, 5: 3, 7: 4}

    pred_classes = preds.cpu().numpy()
    pred_mapped = [predicted_rating_map[p] for p in pred_classes]
    

    # Load metadata
    survey_meta = pd.read_csv(survey_meta_path)
    survey_meta = survey_meta.reset_index(drop=True)  # Make sure indexes align

    survey_meta['predicted_class'] = pred_classes
    survey_meta['predicted_rating_mapped'] = pred_mapped
    survey_meta['difficulty_rating'] = survey_meta['difficulty_rating'].map(difficulty_rating_map)

    # Optional match check
    if 'difficulty_rating' in survey_meta.columns:
        survey_meta['Match'] = survey_meta['difficulty_rating'] == survey_meta['predicted_rating_mapped']

    # Show summary
    print(survey_meta[['trail_id', 'name', 'difficulty_rating', 'gpt_rating', 'predicted_rating_mapped', 'Match']])

    return survey_meta
