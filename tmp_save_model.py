import tensorflow as tf
import mlflow
import mlflow.keras

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"  # If using Docker Desktop on Mac/Windows
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "train_image_model"
try:
    experiment = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    # If experiment already exists, set it
    mlflow.set_experiment(experiment_name)

model_path = "/workspace/models/image_model_MobileNet.keras"
model = tf.keras.models.load_model(model_path)

with mlflow.start_run() as run:  # Start MLflow run
    mlflow.set_tag("Training Info", "Initial Image Model MobileNet")

    mlflow.keras.log_model(
        model, 
        artifact_path="image_classification_model"
    )
    
    print(f"Training completed. Model and metrics logged in MLflow.")

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/image_classification_model"
    model_name = "ImageClassificationModel"
    mlflow.register_model(model_uri, model_name)

# Transition the model to production stage
client = mlflow.tracking.MlflowClient()
latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
client.transition_model_version_stage(
    name=model_name,
    version=latest_version_info.version,
    stage="Production"
)

print(f"Model version {latest_version_info.version} is now in Production stage.")
