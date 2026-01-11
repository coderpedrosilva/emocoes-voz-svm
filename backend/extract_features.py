import os, librosa, numpy as np, pandas as pd

AUDIO_PATH = "../AudioWAV"
OUTPUT = "data/features.csv"

MAP = {
    "HAP": "feliz",
    "SAD": "triste",
    "NEU": "neutro",
    "ANG": "irritado"
}

os.makedirs("data", exist_ok=True)

def extract(file):
    y, sr = librosa.load(file, sr=16000, mono=True)
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

rows = []

for f in os.listdir(AUDIO_PATH):
    if f.endswith(".wav"):
        parts = f.split("_")
        if len(parts) > 2 and parts[2] in MAP:
            emo = MAP[parts[2]]
            feats = extract(os.path.join(AUDIO_PATH, f))
            if feats is not None:
                rows.append(np.hstack([feats, emo]))

cols = [f"mfcc{i}" for i in range(20)] + [f"d{i}" for i in range(20)] + ["zcr","rms","emotion"]
pd.DataFrame(rows, columns=cols).to_csv(OUTPUT, index=False)
print("✔ Features V3 generated.")
