import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from dataloaders import ImagePreprocessor
import os

def evaluate_model(model_weights, test_csv):
    model = Model(img_model_weights=model_weights)
    preprocessor = ImagePreprocessor(train_csv=None, val_csv=test_csv, batch_size=8)
    test_generator = preprocessor.get_val_generator()
    
    true_labels = test_generator.classes
    class_indices = test_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    print(class_labels)
    
    y_pred_prob = model.img_model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(true_labels, y_pred)
    report = classification_report(true_labels, y_pred, target_names=[class_labels[i] for i in range(len(class_labels))])
    conf_matrix = confusion_matrix(true_labels, y_pred)

    # Start MLflow run
    with mlflow.start_run():
        # Log model
        mlflow.keras.log_model(model.img_model, "model")

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Log classification report
        report_df = pd.DataFrame(classification_report(true_labels, y_pred, target_names=[class_labels[i] for i in range(len(class_labels))], output_dict=True)).transpose()
        report_csv_path = "/tmp/classification_report.csv"
        report_df.to_csv(report_csv_path)
        mlflow.log_artifact(report_csv_path)

        # Log confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels.values(), yticklabels=class_labels.values())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        conf_matrix_path = "/tmp/confusion_matrix.png"
        plt.savefig(conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path)

        # Display results
        print(f"\n✅ Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:\n", report)
        plt.show()

    return acc, report, conf_matrix

if __name__ == "__main__":
    model_path = "/workspace/models/image_model_MobileNet.keras"  # ou "models/my_model.h5"
    test_csv = "/workspace/data/processed/test_data.csv"
    
    evaluate_model(model_path, test_csv=test_csv)