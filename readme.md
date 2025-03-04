# Plan de Projet : Monitoring et API de Classification pour Rakuten

## Structure du Projet
```
rakuten-classification/
├── data/
│   ├── raw/                 # Données brutes (X_data.csv, Y_data.csv, images)
│   ├── processed/           # Données prétraitées (splits, features, etc.)
│   ├── unseen/              # Données non vues pour simulation des requêtes utilisateur
│   ├── splits/              # Données splitées (train/test/validation)
│   ├── models/              # Modèles entraînés (text_model.pkl, image_model.pkl)
├── src/
│   ├── preprocessing/       # Scripts de nettoyage, augmentation et préparation des données
│   ├── models/              # Scripts d'entraînement et d'inférence
│   ├── api/                 # Code de l'API FastAPI
│   ├── monitoring/          # Intégration de MLflow pour le suivi des expériences
│   ├── simulation/          # Scripts de simulation de requêtes utilisateur
│   ├── utils/               # Fonctions utilitaires
├── notebooks/               # Notebooks pour l'exploration, tests et analyses
├── docker/
│   ├── Dockerfile           # Image Docker pour l'API
│   ├── docker-compose.yml   # Configuration pour les conteneurs multiples
├── mlflow/                  # Artifacts et logs de MLflow
├── dvc.yaml                 # Pipeline DVC pour gérer les étapes de traitement
├── dagshub/                 # Intégration pour suivi avec DagsHub
├── requirements.txt         # Dépendances Python
├── README.md                # Documentation du projet
```

## Étapes du Projet

### 1. Préparation des Données
1. Importer et organiser les données dans `data/raw`.
2. Nettoyer et prétraiter les données (textes et images).
3. Spliter les données en train/val/test et une partition "non vue".
4. Utiliser DVC pour gérer les versions des données et transformations.

### 2. Développement des Modèles
1. Entraîner ou charger le modèle textuel (text_model.pkl).
2. Entraîner ou charger le modèle visuel (image_model.pkl).
3. Implémenter la fusion des prédictions des deux modèles.
4. Suivre les expérimentations avec MLflow.

### 3. Développement de l'API avec FastAPI
1. Créer les endpoints principaux :
   - **POST /predict** : prédiction sur une description/designation et/ou image.
   - **POST /feedback** : soumettre une correction utilisateur.
   - **POST /retrain** : lancer un entraînement sur les données mises à jour.
2. Ajouter un système de gestion des droits utilisateurs.
3. Tester l'API avec des tests unitaires et d'intégration.
4. Containeriser l'API avec Docker et configurer `docker-compose.yml`.

### 4. Simulation des Requêtes Utilisateurs
1. Écrire un script pour simuler des requêtes fictives à l'API.
2. Utiliser les données "non vues" pour évaluer les performances en conditions réelles.

### 5. Pipeline d'Entraînement et de Déploiement
1. Automatiser les étapes avec DVC (prétraitement, entraînement, sauvegarde des artefacts).
2. Intégrer DVC et MLflow avec DagsHub pour le suivi.
3. Déployer l'API sur un serveur (AWS, Azure, etc.).
4. Configurer CI/CD avec GitHub Actions pour les tests et redéploiements automatiques.

