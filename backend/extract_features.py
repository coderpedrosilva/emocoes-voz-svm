import os, librosa, numpy as np, pandas as pd

AUDIO_PATH = "../AudioWAV"
OUTPUT = "data/features.csv"
os.makedirs("data", exist_ok=True)

MAP = {"HAP":"feliz","SAD":"triste","NEU":"neutro","ANG":"irritado"}

PERC = [10,25,50,75,90]

def stats(X):
    return np.hstack([np.percentile(X, p, axis=1) for p in PERC])

def extract(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if np.mean(librosa.feature.rms(y=y)) < 0.01: return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return np.hstack([
        stats(mfcc), stats(d1), stats(d2),
        np.percentile(zcr, PERC), np.percentile(rms, PERC)
    ])

rows=[]
for f in os.listdir(AUDIO_PATH):
    if f.endswith(".wav"):
        code=f.split("_")[2]
        if code in MAP:
            v=extract(os.path.join(AUDIO_PATH,f))
            if v is not None:
                rows.append(np.hstack([v, MAP[code]]))

cols=[f"f{i}" for i in range(len(rows[0])-1)]+["emotion"]
pd.DataFrame(rows,columns=cols).to_csv(OUTPUT,index=False)
print("✔ V4 features generated.")
