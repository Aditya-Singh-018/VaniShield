import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# LAYER 1 FUNCTION
# -----------------------------
def detect_fake(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    score = (centroid / 5000 + bandwidth / 3000 + zcr * 10) / 3
    fake_score = min(max(score, 0), 1)

    label = "FAKE" if fake_score > 0.5 else "REAL"

    return {
        "fake_score": round(fake_score, 3),
        "label": label
    }


# -----------------------------
# TEST SINGLE FILE
# -----------------------------
print(detect_fake("fake1.wav"))


# -----------------------------
# FILE LIST (CHANGE THESE)
# -----------------------------
real_files = ["real1.wav", "real2.wav"]
fake_files = ["fake1.wav", "fake2.wav"]


# -----------------------------
# GENERATE SCORES
# -----------------------------
real_scores = []
fake_scores = []

# Process real audio
for f in real_files:
    if os.path.exists(f):
        result = detect_fake(f)
        real_scores.append(result["fake_score"])
    else:
        print(f"{f} not found")

# Process fake audio
for f in fake_files:
    if os.path.exists(f):
        result = detect_fake(f)
        fake_scores.append(result["fake_score"])
    else:
        print(f"{f} not found")


# -----------------------------
# COMBINE DATA
# -----------------------------
labels = ["S1", "S2","S3", "S4"]
scores = real_scores + fake_scores

print("Labels:", len(labels))
print("Scores:", len(scores))


# -----------------------------
# SAFETY CHECK
# -----------------------------
if len(labels) != len(scores):
    print("ERROR: mismatch between labels and scores")
    exit()


# -----------------------------
# GRAPH
# -----------------------------
plt.figure()

# Add colors (IMPORTANT)
colors = ["green" if s <= 0.5 else "red" for s in scores]
bars = plt.bar(labels, scores, color=colors)

# Add Legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='green', label='Real'),
    Patch(facecolor='red', label='Fake')
]

plt.legend(handles=legend_elements)

# Value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.2f}', ha='center', va='bottom')

# Threshold line + label
plt.axhline(y=0.5, linestyle='--')
plt.text(0, 0.52, "Threshold = 0.5", fontsize=9)

# Title
plt.title("Fake vs Real Audio Detection (Layer 1)")

# Axis labels
plt.xlabel("Samples")
plt.ylabel("Fake Score")

plt.savefig("graph.png")
plt.show()