import os
from ultralytics import YOLO

def train_custom_model(data_path='data.yaml', model_path='yolov8s.pt', epochs=100, imgsz=640):
    """
    Trains a YOLOv8 model on a custom dataset with improved parameters.
    data_path: Path to the data.yaml file defining the dataset.
    """
    print(f"Starting training with model: {model_path} on {data_path}...")
    
    # Load a model
    model = YOLO(model_path)

    # Train the model with some robust settings
    try:
        results = model.train(
            data=data_path, 
            epochs=epochs, 
            imgsz=imgsz, 
            patience=20,  # Early stopping if no improvement
            batch=8,      # Small batch size for better convergence on small datasets
            name='improved_fruit_classifier'
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    train_custom_model()
