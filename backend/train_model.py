import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump

df = pd.read_csv("data/features.csv")
X, y = df.drop("emotion",axis=1), df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=30, gamma="scale", probability=True))
])

model.fit(X_train,y_train)
print(classification_report(y_test, model.predict(X_test)))

os.makedirs("model",exist_ok=True)
dump(model,"model/svm_emotions.joblib")
print("✔ Model V3 trained.")
