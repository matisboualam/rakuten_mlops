import os
import numpy as np
from src.modeling.dataloaders import ImagePreprocessor

# Simulation de fichiers CSV (tu peux mettre de vrais chemins)
csv = "/workspace/data/processed/val.csv"

# Vérifie si les fichiers existent
if not os.path.exists(csv):
    print("Les fichiers CSV n'existent pas. Assurez-vous qu'ils sont bien placés.")
else:
    # Instanciation de la classe
    preprocessor = ImagePreprocessor(csv, image_size=(224, 224), batch_size=8)

    # Test des générateurs
    try:
        generator = preprocessor.get_generator()
        
        # Affiche quelques informations pour vérifier
        print(f"Nombre de classes détectées (train): {len(generator.class_indices)}")

        # Affichage d'un batch d'exemple
        batch = next(generator)
        print(f"Batch shape (images): {batch[0].shape}")
        print(f"Batch shape (labels): {batch[1].shape}")

        # random_indices = np.random.choice(batch.shape[0], 5, replace=False)
        
        # for idx in random_indices:
        #     img = batch_images[idx]
            
        #     # Check image shape
        #     if img.shape != (224, 224, 3):
        #         print(f"⚠️ Image at index {idx} has an incorrect shape: {img.shape}")
        #     else:
        #         print(f"✅ Image at index {idx} has correct shape: {img.shape}")

        #     # Check normalization (values should be between 0 and 1)
        #     if np.min(img) < 0 or np.max(img) > 1:
        #         print(f"⚠️ Image at index {idx} is NOT normalized correctly (min: {np.min(img)}, max: {np.max(img)})")
        #     else:
        #         print(f"✅ Image at index {idx} is correctly normalized (min: {np.min(img)}, max: {np.max(img)})")
    
    except Exception as e:
        print(f"Erreur lors de la génération des données : {e}")
