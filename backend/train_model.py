import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump

df=pd.read_csv("data/features.csv")
X,y=df.drop("emotion",axis=1),df["emotion"]

Xtr,Xts,Ytr,Yts=train_test_split(X,y,test_size=0.2,stratify=y)

model=Pipeline([
    ("scaler",StandardScaler()),
    ("svm",SVC(C=50,kernel="rbf",gamma="scale",probability=True))
])

model.fit(Xtr,Ytr)
print(classification_report(Yts,model.predict(Xts)))

os.makedirs("model",exist_ok=True)
dump(model,"model/svm_emotions.joblib")
print("✔ V4 SVM trained.")
