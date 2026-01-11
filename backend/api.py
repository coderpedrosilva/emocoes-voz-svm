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

def extract_features(file_bytes):
    y, sr = librosa.load(file_bytes, sr=16000, mono=True)

    # filtro de silêncio
    if np.mean(librosa.feature.rms(y=y)) < 0.01:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return np.hstack([
        mfcc.mean(axis=1),
        delta.mean(axis=1),
        zcr.mean(),
        rms.mean()
    ])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    feats = extract_features(io.BytesIO(data))

    if feats is None:
        return {"emotion": "silence", "confidence": 1.0}

    feats = feats.reshape(1, -1)
    pred = MODEL.predict(feats)[0]
    prob = MODEL.predict_proba(feats).max()

    return {"emotion": pred, "confidence": float(prob)}
