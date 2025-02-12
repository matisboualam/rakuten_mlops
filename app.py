import pandas as pd
from fastapi import FastAPI

from src.modeling.models import Model

# Définition de l'application FastAPI
app = FastAPI()

MLFLOW_TRACKING_URI = "/workspace/mlruns/2/5f23b7c819544fc49e6fb53206cad84a/artifacts/image_classification_model/"

print(f"Loading model ...")
model = Model(img_model_weights=MLFLOW_TRACKING_URI)
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
            "image_path": image_path,
            "predicted_class": str(prediction)
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/feedback")
async def feedback(indice: int, correct: bool, annotation: str = None):
    """Enregistre le feedback de l'utilisateur dans un fichier CSV."""
    try:
        global train_data, user_data
        input = user_data.iloc[indice]

        # Add the input to train_data but do not modify user_data
        if correct:
            print(len(train_data))
            # Use concat instead of append
            train_data = pd.concat([train_data, input.to_frame().T], ignore_index=True)
            print(len(train_data))
            
        else:
            if annotation is not None:
                input['prdtypecode'] = annotation
            print(len(train_data))
            # Use concat instead of append
            train_data = pd.concat([train_data, input.to_frame().T], ignore_index=True)
            print(len(train_data))

        # Optionally, remove the entry from the user_data if needed
        user_data.drop(indice, inplace=True)  # Uncomment if you want to remove the row from user_data
        train_data.to_csv('data/processed/data.csv', index=False)
        user_data.to_csv('data/processed/unseen.csv', index=False)
        

        return {"message": "Feedback recorded successfully."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
