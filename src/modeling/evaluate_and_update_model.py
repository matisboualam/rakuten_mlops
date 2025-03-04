import mlflow
import mlflow.keras

def evaluate_and_update_model(**context):
    """Évalue les différents runs et met à jour le modèle en fonction de l'accuracy"""
    
    # S'assurer que le serveur MLflow est accessible
    if mlflow.get_tracking_uri() is None:
        raise ValueError("MLflow tracking URI n'est pas défini. Assurez-vous que MLflow est correctement configuré.")
    
    # Recherche de tous les runs MLflow pour le type 'text_model'
    runs = mlflow.search_runs(
        filter_string="tags.model_type = 'text_model'",  # Filtrer par le type de modèle
        order_by=["accuracy desc"],  # Trier par accuracy décroissante
        max_results=5  # Limiter à un certain nombre de résultats
    )

    if runs.empty:
        raise ValueError("Aucun run trouvé avec le type de modèle 'text_model'.")

    # Sélectionner le run avec la meilleure accuracy
    best_run = runs.iloc[0]

    # Extraire les informations du meilleur modèle
    best_run_id = best_run.run_id
    best_accuracy = best_run.accuracy
    best_val_accuracy = best_run.val_accuracy

    # Charger le modèle du meilleur run
    model_uri = f"runs:/{best_run_id}/text_model"
    best_model = mlflow.keras.load_model(model_uri)

    # Logique pour la mise à jour du modèle
    threshold_accuracy = 0.85  # Seuil pour mettre à jour le modèle

    if best_accuracy > threshold_accuracy:
        context['ti'].xcom_push(key='best_model', value="new")
        print("Le modèle sera mis à jour.")
    else:
        context['ti'].xcom_push(key='best_model', value="old")
        print("Le modèle actuel reste en place.")

    return "Model evaluation completed and best model selected."
