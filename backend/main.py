import subprocess

print("Starting emocoes-voz-svm ML pipeline...\n")

print("Step 1: Extracting audio features")
subprocess.call(["python", "extract_features.py"])

print("\nStep 2: Training SVM model")
subprocess.call(["python", "train_model.py"])

print("\nPipeline finished successfully.")
