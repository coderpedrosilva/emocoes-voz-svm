from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa, joblib, os, io

app = FastAPI(title="Voice Emotion SVM")

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
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    return np.hstack([mfcc.mean(axis=1), zcr.mean(), rms.mean(), pitch.mean()])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    feats = extract_features(io.BytesIO(data)).reshape(1,-1)
    pred = MODEL.predict(feats)[0]
    prob = MODEL.predict_proba(feats).max()
    return {"emotion": pred, "confidence": float(prob)}
