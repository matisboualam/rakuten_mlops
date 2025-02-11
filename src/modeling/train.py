import os
import tensorflow as tf
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models import Model

MLFLOW_TRACKING_URI = "http://localhost:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("image_classification")

def train_image_model(model_path, train_data_path, val_data_path, test_data_path=None, nb_epochs=1):
    """Train an image classification model and track performance with MLflow."""
    
    model = Model(img_model_weights=model_path)

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print(f"Error: Training or validation data not found.")
        return
    
    train_data_gen = model.img_dataloader.get_generator(train_data_path)
    val_data_gen = model.img_dataloader.get_generator(val_data_path)

    steps_per_epoch = train_data_gen.samples // model.img_dataloader.batch_size
    validation_steps = val_data_gen.samples // model.img_dataloader.batch_size

    print(f"Using model from: {model_path}")
    print(f"Training on: {train_data_path}, Validation on: {val_data_path}")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Using device: {device}")

    with mlflow.start_run():  # Start MLflow run
        mlflow.log_param("batch_size", model.img_dataloader.batch_size)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "categorical_crossentropy")
        mlflow.log_param("epochs", nb_epochs)

        # Training
        with tf.device(device):
            model.img_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.img_model.fit(
                train_data_gen, 
                # steps_per_epoch=steps_per_epoch, 
                validation_data=val_data_gen,
                # validation_steps=validation_steps,
                epochs=nb_epochs
            )

        # Log metrics
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)

        # Test evaluation
        if not os.path.exists(test_data_path):
            print(f"Error: Test data not found at {test_data_path}")
            return
        
        test_data_gen = model.img_dataloader.get_generator(test_data_path)
        true_labels = test_data_gen.classes
        class_indices = test_data_gen.class_indices
        class_labels = {v: k for k, v in class_indices.items()}
        
        y_pred_prob = model.img_model.predict(test_data_gen)
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(true_labels, y_pred)
        report = classification_report(true_labels, y_pred, target_names=[class_labels[i] for i in range(len(class_labels))])

        # Log test metrics
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_text(report, "classification_report.txt")

        # Prepare model logging
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Adjust shape as needed
        input_example = tf.convert_to_tensor(dummy_input)

        # Save model in the registry
        mlflow.keras.log_model(model.img_model, "image_classification_model")
        print(f"Training completed. Model and metrics logged in MLflow.")

if __name__ == "__main__":
    model_path = "/workspace/models/image_model_MobileNet.keras"
    train_data_path = "/workspace/data/processed/train.csv"
    val_data_path = "/workspace/data/processed/val.csv"
    test_data_path = "/workspace/data/processed/test.csv"
    nb_epochs = 10

    train_image_model(model_path, train_data_path, val_data_path, test_data_path, nb_epochs)
