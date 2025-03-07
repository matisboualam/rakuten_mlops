from src.modeling.models import Model

def train_model():
    model = Model(txt_model_weights="workspace/models/textmodel_27_classes.keras")  # Remplacez par le bon chemin
    model.train_text()
    print("Model training completed.")

if __name__ == "__main__":
    train_model()
