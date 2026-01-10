import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from tqdm import tqdm

# garante que a pasta model/ exista
os.makedirs("model", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/features.csv")

X = df.drop("emotion", axis=1)
y = df["emotion"]

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training SVM model...")

model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)

for _ in tqdm(range(1), desc="Training progress"):
    model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

dump(model, "model/svm_emotions.joblib")
print("✔ Model saved in backend/model/")
