import os, librosa, numpy as np, pandas as pd

AUDIO_PATH = "../AudioWAV"
OUTPUT = "data/features.csv"

MAP = {
    "ANG": "irritado",
    "HAP": "feliz",
    "SAD": "triste",
    "FEA": "ansioso",
    "DIS": "irritado",
    "NEU": "neutro"
}

# garante que a pasta data/ exista
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

def extract(file):
    y, sr = librosa.load(file, sr=16000, mono=True, duration=2.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=256)
    rms = librosa.feature.rms(y=y, hop_length=256)
    pitch = librosa.yin(y, fmin=50, fmax=300)

    return np.hstack([mfcc.mean(axis=1), zcr.mean(), rms.mean(), pitch.mean()])

rows = []

files = [f for f in os.listdir(AUDIO_PATH) if f.endswith(".wav")]

for i, f in enumerate(files):
    parts = f.split("_")
    if len(parts) > 2 and parts[2] in MAP:
        emotion = MAP[parts[2]]
        path = os.path.join(AUDIO_PATH, f)
        rows.append(np.hstack([extract(path), emotion]))

    if i % 200 == 0:
        print(f"Processed {i}/{len(files)}")

df = pd.DataFrame(rows, columns=[f"mfcc{i}" for i in range(13)] + ["zcr","rms","pitch","emotion"])
df.to_csv(OUTPUT, index=False)

print("✔ Feature dataset generated.")
