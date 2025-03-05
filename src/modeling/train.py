import os
import tensorflow as tf
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from models import Model

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"  # If using Docker Desktop on Mac/Windows
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "train_image_model"
try:
    experiment = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    # If experiment already exists, set it
    mlflow.set_experiment(experiment_name)


def train_image_model(model_path, train_data_path, val_data_path, test_data_path=None, nb_epochs=1):
    """Train an image classification model and track performance with MLflow."""
    
    model = Model(img_model_weights=model_path)

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path) or not os.path.exists(test_data_path):
        print(f"Error: Training, Validation or Test data not found.")
        return
    
    train_data_gen = model.img_dataloader.get_generator(train_data_path)
    val_data_gen = model.img_dataloader.get_generator(val_data_path)
    test_data_gen = model.img_dataloader.get_generator(test_data_path)

    params = {
        "batch_size": model.img_dataloader.batch_size,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "epochs": nb_epochs
    }
    print(params)

    model.img_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.img_model.fit(
        train_data_gen, 
        validation_data=val_data_gen,
        epochs=nb_epochs
    )

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    true_labels = test_data_gen.classes
    class_indices = test_data_gen.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    y_pred_prob = model.img_model.predict(test_data_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)

    test_acc = accuracy_score(true_labels, y_pred)
    report = classification_report(true_labels, y_pred, target_names=[class_labels[i] for i in range(len(class_labels))])

    with mlflow.start_run():  # Start MLflow run
        mlflow.log_params(params)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_text(report, "classification_report.txt")

        mlflow.set_tag("Training Info", "Image Model", "MobileNet")

        mlflow.keras.log_model(
            model.img_model, 
            artifact_path = "image_classification_model"
            )
        
        print(f"Training completed. Model and metrics logged in MLflow.")

if __name__ == "__main__":
    model_path = "/workspace/models/image_model_MobileNet.keras"
    train_data_path = "/workspace/data/processed/train.csv"
    val_data_path = "/workspace/data/processed/val.csv"
    test_data_path = "/workspace/data/processed/test.csv"
    nb_epochs = 1

    train_image_model(model_path, train_data_path, val_data_path, test_data_path, nb_epochs)
