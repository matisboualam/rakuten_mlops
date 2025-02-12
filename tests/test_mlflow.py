import mlflow.keras
from src.modeling.models import Model


MLFLOW_TRACKING_URI = "/workspace/mlruns/2/5f23b7c819544fc49e6fb53206cad84a/artifacts/image_classification_model/"


model_test = mlflow.keras.load_model(MLFLOW_TRACKING_URI)

model = Model(img_model_weights=MLFLOW_TRACKING_URI)

print(model.predict_img('/workspace/data/raw/img/image_1269950347_product_3964906385.jpg'))

print("Model loaded successfully.")

