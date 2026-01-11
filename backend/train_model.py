import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump
from tqdm import tqdm
import os

print("Loading dataset...")
df = pd.read_csv("data/features.csv")

X = df.drop("emotion", axis=1)
y = df["emotion"]

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training balanced SVM...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"))
])

for _ in tqdm(range(1), desc="Training"):
    model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
dump(model, "model/svm_emotions.joblib")
print("✔ Model V2 saved.")
