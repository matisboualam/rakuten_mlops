from models import Model
import mlflow
import mlflow.keras 

mlflow.set_experiment("image_model")
mlflow.keras.autolog()

model_path = "/workspace/models/image_model_MobileNet.keras"

def train_image_model(model_path):

    classifier = Model(img_model_weights=model_path)

    classifier.train_img_model()

    mlflow.keras.log_model(classifier.img_model, "model")

    print("Model trained and logged in MLflow!")