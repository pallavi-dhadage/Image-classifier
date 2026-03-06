import os
import subprocess
import sys

def run_project():
    print("🚀 Initializing AI Image Classifier Project...")

    # 1. Check directories
    folders = [
        os.path.join('static', 'uploads'),
        os.path.join('static', 'results'),
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

    # 2. Check model weights
    model_weights = 'runs/detect/improved_fruit_classifier/weights/best.pt'
    if not os.path.exists(model_weights):
        print("⚠️ Warning: Improved weights not found. Falling back to default or needing training.")
        # If no weights, maybe ask the user to train or provide a default
    else:
        print(f"✅ Found improved weights: {model_weights}")

    # 3. Start the Flask server
    print("\n--- Starting Application ---")
    print("🔗 Visit http://127.0.0.1:5000 to use the application.")
    print("Press CTRL+C to stop the server.\n")

    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except Exception as e:
        print(f"❌ Error starting the application: {e}")

if __name__ == "__main__":
    run_project()
