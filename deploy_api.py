import pandas as pd
import os
import atexit
from fastapi import FastAPI
from src.modeling.models import Model

# Définition de l'application FastAPI
deploy_api = FastAPI()

model_weights = "models:/ImageClassificationModel/1"

print(f"Loading model ...")
model = Model(
    img_model_weights=model_weights
    )
print("Model loaded successfully.")

user_data = pd.read_csv('data/processed/unseen.csv')
train_data = pd.read_csv('data/processed/data.csv')

# Create tmp_predicted_data.csv file
tmp_predicted_data_path = 'data/processed/tmp/tmp_predicted_data.csv'
# Create an empty tmp_predicted_data.csv file
pd.DataFrame(columns=['image_path', 'description', 'prdtypecode']).to_csv(tmp_predicted_data_path, index=False)

# Register atexit function to delete the file when the API is closed
def cleanup():
    if os.path.exists(tmp_predicted_data_path):
        os.remove(tmp_predicted_data_path)
        print(f"{tmp_predicted_data_path} deleted successfully.")

atexit.register(cleanup)

@deploy_api.post("/predict")
async def predict(indice: int):
    """Prend un chemin d'image en entrée et retourne la classe prédite."""
    try:
        global train_data, user_data
        input = user_data.iloc[indice]
        image_path = input['image_path']
        text = input['description']

        # Directly pass the image path to your model (since it handles preprocessing)
        prediction = model.predict_img(image_path)  # Ensure it's a list if needed
        
        return {
            "indice": indice,
            "image_path": image_path,
            "predicted_class": str(prediction)
        }
    
    except Exception as e:
        return {"error": str(e)}

@deploy_api.post("/predict/feedback")
async def feedback(indice: int, predicted_class: str):
    """Enregistre le feedback de l'utilisateur dans un fichier CSV."""
    try:
        global train_data, user_data
        input = user_data.iloc[indice]

        feedback_message = ""
        if input['prdtypecode'] == predicted_class:
            feedback_message = '✅ Correct prediction!'
        else:
            feedback_message = '⚠️ Incorrect prediction!'

        # Check if the data is already in the temporary CSV file
        tmp_data = pd.read_csv(tmp_predicted_data_path)
        if not ((tmp_data['image_path'] == input['image_path']) & (tmp_data['description'] == input['description'])).any():
            # Append the feedback data to the temporary CSV file
            input.to_frame().T.to_csv(tmp_predicted_data_path, mode='a', header=False, index=False)

        return {
            "message": "Feedback recorded successfully.",
            "feedback": feedback_message,
            "user_data_size": len(user_data)
        }
    except Exception as e:
        return {"error": str(e)}
    
@deploy_api.post("/merge")
async def merge_data():
    """Merge tmp_predicted_data with data.csv while avoiding duplicates and clean tmp csv."""
    try:
        global train_data

        # Read the temporary predicted data
        tmp_data = pd.read_csv(tmp_predicted_data_path)

        # Print the length of the original data
        original_length = len(train_data)
        print(f"Original data length: {original_length}")

        # Concatenate the dataframes and drop duplicates
        combined_data = pd.concat([train_data, tmp_data]).drop_duplicates(subset=['image_path', 'description'])

        # Save the combined data back to the main data file
        combined_data.to_csv('data/processed/data.csv', index=False)

        # Update the global train_data with the combined data
        train_data = combined_data

        # Print the length of the combined data
        combined_length = len(combined_data)
        print(f"Combined data length: {combined_length}")

        # Clean the temporary CSV file
        pd.DataFrame(columns=['image_path', 'description', 'prdtypecode']).to_csv(tmp_predicted_data_path, index=False)

        return {"message": "Data merged and temporary file cleaned successfully.", "original_length": original_length, "combined_length": combined_length}
    except Exception as e:
        return {"error": str(e)}