from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa, joblib, io

app = FastAPI(title="Voice Emotion SVM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API running"}

MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = joblib.load("model/svm_emotions.joblib")
    print("✔ Model loaded")


# =================== V4 SCIENTIFIC FEATURE PIPELINE ===================

def extract_features(fileobj):
    y, sr = librosa.load(fileobj, sr=16000, mono=True)

    # Silence filter
    if np.mean(librosa.feature.rms(y=y)) < 0.01:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def stats(X):
        return np.hstack([
            np.percentile(X, 10, axis=1),
            np.percentile(X, 25, axis=1),
            np.percentile(X, 50, axis=1),
            np.percentile(X, 75, axis=1),
            np.percentile(X, 90, axis=1),
        ])

    return np.hstack([
        stats(mfcc),
        stats(d1),
        stats(d2),
        np.percentile(zcr, [10,25,50,75,90]),
        np.percentile(rms, [10,25,50,75,90])
    ])

# =====================================================================


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    feats = extract_features(io.BytesIO(data))

    if feats is None:
        return {"emotion": "silence", "confidence": 1.0}

    feats = feats.reshape(1, -1)
    pred = MODEL.predict(feats)[0]
    prob = float(MODEL.predict_proba(feats).max())

    return {"emotion": pred, "confidence": prob}
