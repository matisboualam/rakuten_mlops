import pandas as pd
from fastapi import FastAPI

from src.modeling.models import Model

# Définition de l'application FastAPI
app = FastAPI()

model_weights = "/workspace/models/image_model_MobileNet.keras"
MLFLOW_TRACKING_URI = "/workspace/mlruns/2/5f23b7c819544fc49e6fb53206cad84a/artifacts/image_classification_model/"

print(f"Loading model ...")
model = Model(
    img_model_weights=model_weights,
    # mlflow_img_model_weights=MLFLOW_TRACKING_URI
    )
print("Model loaded successfully.")

user_data = pd.read_csv('data/processed/unseen.csv')
train_data = pd.read_csv('data/processed/data.csv')

@app.post("/predict")
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

@app.post("/predict/feedback")
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

        train_data = pd.concat([train_data, input.to_frame().T], ignore_index=True)
        user_data.drop(indice, inplace=True)

        train_data.to_csv('data/processed/data.csv', index=False)
        user_data.to_csv('data/processed/unseen.csv', index=False)

        return {
            "message": "Feedback recorded successfully.",
            "feedback": feedback_message,
            "train_data_new_size": len(train_data),
            "user_data_new_size": len(user_data)
        }
    except Exception as e:
        return {"error": str(e)}

