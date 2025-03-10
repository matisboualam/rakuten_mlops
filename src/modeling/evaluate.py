import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from models import Model  # Assurez-vous que le modèle est correctement importé
from tensorflow.keras.models import load_model
from dataloaders import ImagePreprocessor

def get_latest_model_uri(model_name="ImageClassificationModel", stage="Production"):
    """
    Récupère l'URI du modèle depuis MLflow selon le stage spécifié.
    
    Args:
        model_name (str): Nom du modèle enregistré dans MLflow.
        stage (str): Stage du modèle ("Production" ou "Staging").
    
    Returns:
        str: URI du modèle ou None si non trouvé.
    """
    MLFLOW_TRACKING_URI = "http://modeling:5000"  # Si Docker Desktop sur Mac/Windows
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.MlflowClient()
    model_versions = client.get_latest_versions(model_name, stages=[stage])
    
    if model_versions:
        model_version = model_versions[0].version
        return f"models:/{model_name}/{model_version}"
    return None

def get_latest_trained_model():
    """
    Récupère l'URI du dernier modèle entraîné dans la dernière run de MLflow.
    
    Returns:
        str: URI du modèle ou None si non trouvé.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("train_image_model")
    
    if not experiment:
        return None
    
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
    
    if runs:
        run_id = runs[0].info.run_id
        run_uri = f"runs:/{run_id}/image_classification_model"
        return run_uri
    return None

def evaluate_model(model_uri, test_data_path="test.csv"):
    """
    Charge un modèle et l'évalue sur les données de test.
    
    Args:
        model_uri (str): URI du modèle à évaluer.
        test_data_path (str): Chemin vers les données de test.
    
    Returns:
        dict: Résultats de l'évaluation (accuracy, classification report)
    """
    print(f"Loading model from URI: {model_uri}")
    model = Model(img_model_weights=model_uri)  # Chargement du modèle MLflow
    print("Model loaded!")
    
    # Charger les données de test
    print(f"Loading test data from {test_data_path}")
    data_gen = ImagePreprocessor()
    test_data_gen = data_gen.get_generator(test_data_path)
    true_labels = test_data_gen.classes
    class_indices = test_data_gen.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Effectuer des prédictions
    print("Making predictions on test data...")
    y_pred_prob = model.img_model.predict(test_data_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcul de l'accuracy et du rapport de classification
    test_acc = accuracy_score(true_labels, y_pred)
    report = classification_report(true_labels, y_pred, target_names=[class_labels[i] for i in range(len(class_labels))])
    
    return {"accuracy": test_acc, "classification_report": report}

def compare_models():
    """
    Compare le dernier modèle en production avec le dernier modèle entraîné.
    Si le modèle entraîné est meilleur, il est enregistré en tant que modèle de production.
    """
    production_model_uri = get_latest_model_uri()
    trained_model_uri = get_latest_trained_model()
    
    if not production_model_uri or not trained_model_uri:
        print("Impossible de récupérer les modèles à comparer.")
        return
    
    print("Evaluating production model...")
    print("URI:", production_model_uri)
    prod_results = evaluate_model(production_model_uri, test_data_path="/workspace/data/processed/test.csv")
    print("Production Model Accuracy:", prod_results["accuracy"])
    print("Production Classification Report:\n", prod_results["classification_report"])
    
    print("Evaluating latest trained model...")
    print("URI:", trained_model_uri)
    trained_results = evaluate_model(trained_model_uri, test_data_path="/workspace/data/processed/test.csv")
    print("Latest Trained Model Accuracy:", trained_results["accuracy"])
    print("Latest Trained Classification Report:\n", trained_results["classification_report"])
    
    if trained_results["accuracy"] > prod_results["accuracy"]:
        print("The latest trained model is better than the production model. Promoting to production...")
        model_name = "ImageClassificationModel"
        mlflow.register_model(trained_model_uri, model_name)
        print("Model promoted to production.")
    else:
        print("The production model is still better. No changes made.")
    
    print("Comparison Done!")

# Lancer la comparaison des modèles
compare_models()
