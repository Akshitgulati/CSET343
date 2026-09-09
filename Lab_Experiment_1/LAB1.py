# LIBRARIES
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import wfdb
import nltk

from nltk.tokenize import word_tokenize

from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "processed.cleveland.data")

df = pd.read_csv(
    file_path,
    header=None,
    na_values="?"
)


columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df.columns = columns

print("First 5 rows:")
print(df.head())



print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values before preprocessing:")
print(df.isnull().sum())



df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

print("\nMissing values after preprocessing:")
print(df.isnull().sum())



print("\nStatistical summary:")
print(df.describe())



print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

plt.figure(figsize=(8, 5))

df["target"].value_counts().sort_index().plot(kind="bar")

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)

plt.savefig(os.path.join(BASE_DIR, "target_distribution.png"))
plt.show()



correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)



plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(os.path.join(BASE_DIR, "correlation_heatmap.png"))
plt.show()



X = df.drop("target", axis=1)

y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nOriginal target:")
print(y.head())




y = (y > 0).astype(int)

print("\nBinary target distribution:")
print(y.value_counts())





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)




model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)




y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]




accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)




roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC Score:")
print(roc_auc)



cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
).plot()

plt.title("Heart Disease Confusion Matrix")

plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.show()



record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb"
)

annotation = wfdb.rdann(
    "100",
    "atr",
    pn_dir="mitdb"
)

print("\nECG Sampling Frequency:")
print(record.fs)

print("\nECG Signal Shape:")
print(record.p_signal.shape)

print("\nNumber of ECG Annotations:")
print(len(annotation.sample))



signal_data = record.p_signal[:, 0]

num_samples = 2000

plt.figure(figsize=(14, 5))

plt.plot(
    signal_data[:num_samples]
)

plt.title("MIT-BIH ECG Record 100")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.savefig(os.path.join(BASE_DIR, "ecg_signal.png"))
plt.show()



annotation_samples = annotation.sample

selected_annotations = annotation_samples[
    annotation_samples < num_samples
]

if len(selected_annotations) > 1:

    rr_intervals = (
        np.diff(selected_annotations) / record.fs
    )

    mean_rr = np.mean(rr_intervals)

    heart_rate = 60 / mean_rr

    print("\nMean RR Interval:")
    print(mean_rr)

    print("\nApproximate Heart Rate:")
    print(heart_rate, "BPM")

else:

    print("\nNot enough ECG annotations for heart-rate calculation.")



clinical_note = """
Patient complains of chest pain and shortness of breath.
The patient has elevated blood pressure and cholesterol.
ECG examination shows abnormal results.
"""

print("\nClinical Note:")
print(clinical_note)




nltk.download(
    "punkt_tab",
    quiet=True
)

tokens = word_tokenize(
    clinical_note.lower()
)

print("\nTokens:")
print(tokens)



word_frequency = {}

for word in tokens:

    if word.isalpha():

        if word in word_frequency:
            word_frequency[word] += 1

        else:
            word_frequency[word] = 1


print("\nWord Frequency:")

for word, frequency in word_frequency.items():

    print(
        word,
        ":",
        frequency
    )



image = Image.new(
    "L",
    (512, 512),
    0
)

draw = ImageDraw.Draw(image)

draw.ellipse(
    (100, 50, 412, 462),
    fill=80
)

draw.ellipse(
    (150, 100, 250, 420),
    fill=130
)

draw.ellipse(
    (262, 100, 362, 420),
    fill=130
)

draw.ellipse(
    (220, 220, 292, 350),
    fill=180
)



image_path = os.path.join(
    BASE_DIR, "data", "synthetic_medical_image.png"
)

image.save(image_path)

loaded_image = Image.open(
    image_path
)

print("\nOriginal Image Size:")
print(loaded_image.size)

resized_image = loaded_image.resize(
    (224, 224)
)

print("\nResized Image Size:")
print(resized_image.size)



plt.figure(figsize=(6, 6))

plt.imshow(
    resized_image,
    cmap="gray"
)

plt.title("Synthetic Medical Image")

plt.axis("off")

plt.savefig(os.path.join(BASE_DIR, "synthetic_image_display.png"))
plt.show()


print("\nMultimodal Healthcare Data Integration:")

print("""
Healthcare data can contain multiple modalities:

1. Tabular data:
   Patient demographics, blood pressure,
   cholesterol and other clinical measurements.

2. ECG signal data:
   ECG waveforms can provide information
   about heart rhythm and cardiac activity.

3. Textual data:
   Clinical notes can contain symptoms,
   observations and medical history.

4. Image data:
   Medical images can provide visual
   information about a patient's condition.

These different modalities can be combined
using patient IDs and timestamps.

Features extracted from the different modalities
can be used together in a future multimodal
machine learning system.
""")


print("\nLab Experiment 1 completed successfully.")