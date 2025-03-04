import pytest
from unittest.mock import patch
import pandas as pd
from tasks.check_data_growth import check_data_growth
from airflow.models import Variable

@patch("pandas.read_csv")
@patch("airflow.models.Variable.get")
@patch("airflow.models.Variable.set")
def test_check_data_growth(mock_set, mock_get, mock_read_csv):
    # Simuler un fichier CSV avec 20 lignes
    mock_read_csv.return_value = pd.DataFrame({'col1': range(20)})

    # Simuler une ancienne valeur stockée dans Variable (ici "4")
    mock_get.return_value = "4"
    
    # Impression pour voir ce que retourne le mock de Variable
    print("Simulated previous size:", mock_get.return_value)
    
    # Préparer le contexte pour l'exécution de la fonction
    context = {'ti': {'xcom_push': lambda key, value: None}}  # Utilisation du mock xcom_push

    # Appel de la fonction
    result = check_data_growth(**context)
    
    # Impression pour voir le résultat de la fonction
    print("Result from check_data_growth:", result)

    # Vérification du résultat
    assert result == "process_data"  # Il y a plus de 15 nouvelles lignes
    
    # Vérifier que Variable.set a bien été appelé pour stocker la taille du fichier
    mock_set.assert_called_once_with("data_size", 20)  # Vérifier que la taille du fichier a bien été mise à jour
