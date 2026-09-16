


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import wfdb
import scipy.signal

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


#TABULAR DATA--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TABULAR_DIR = os.path.join(BASE_DIR, "data", "tabular")
TABULAR_FILE = os.path.join(TABULAR_DIR, "diabetes.csv")

df = pd.read_csv(TABULAR_FILE)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate records:")
print(duplicate_count)

clinical_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\nZero values in clinical measurements:")

for column in clinical_columns:
    zero_count = (df[column] == 0).sum()
    print(column, ":", zero_count)

df_clean = df.copy()

for column in clinical_columns:
    df_clean[column] = df_clean[column].replace(0, np.nan)

print("\nMissing values after replacing invalid zeros:")
print(df_clean.isnull().sum())

plt.figure(figsize=(12, 6))

sns.heatmap(
    df_clean.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Before Imputation")
plt.xlabel("Features")
plt.ylabel("Patients")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TABULAR_DIR,
        "tabular_missing_values_heatmap.png"
    )
)

plt.show()

for column in clinical_columns:
    median_value = df_clean[column].median()
    df_clean[column] = df_clean[column].fillna(median_value)

print("\nMissing values after median imputation:")
print(df_clean.isnull().sum())

print("\nCleaned dataset:")
print(df_clean.head())

print("\nCleaned dataset shape:")
print(df_clean.shape)

print("\nSummary statistics:")
print(df_clean.describe())

plt.figure(figsize=(12, 8))

correlation_matrix = df_clean.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TABULAR_DIR,
        "tabular_correlation_heatmap.png"
    )
)

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["Glucose"],
    kde=True
)

plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TABULAR_DIR,
        "glucose_distribution.png"
    )
)

plt.show()

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_clean,
    x="BMI",
    y="Glucose",
    hue="Outcome"
)

plt.title("BMI vs Glucose")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TABULAR_DIR,
        "bmi_vs_glucose.png"
    )
)

plt.show()

glucose_diabetic = df_clean[
    df_clean["Outcome"] == 1
]["Glucose"]

glucose_non_diabetic = df_clean[
    df_clean["Outcome"] == 0
]["Glucose"]

t_statistic, p_value = ttest_ind(
    glucose_diabetic,
    glucose_non_diabetic,
    equal_var=False
)

print("\nIndependent T-Test")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("Statistically significant difference between the two groups.")
else:
    print("No statistically significant difference between the two groups.")

df_clean["High_Glucose"] = np.where(
    df_clean["Glucose"] >= 126,
    "High",
    "Normal"
)

contingency_table = pd.crosstab(
    df_clean["High_Glucose"],
    df_clean["Outcome"]
)

print("\nChi-Square Test")
print("\nContingency table:")
print(contingency_table)

chi2, chi_p_value, degrees_of_freedom, expected = chi2_contingency(
    contingency_table
)

print("\nChi-square statistic:", chi2)
print("P-value:", chi_p_value)
print("Degrees of freedom:", degrees_of_freedom)

if chi_p_value < 0.05:
    print(
        "Statistically significant association between "
        "glucose category and outcome."
    )
else:
    print(
        "No statistically significant association detected."
    )

bmi_group_1 = df_clean[
    df_clean["BMI"] < 25
]["Glucose"]

bmi_group_2 = df_clean[
    (df_clean["BMI"] >= 25) &
    (df_clean["BMI"] < 30)
]["Glucose"]

bmi_group_3 = df_clean[
    df_clean["BMI"] >= 30
]["Glucose"]

anova_statistic, anova_p_value = f_oneway(
    bmi_group_1,
    bmi_group_2,
    bmi_group_3
)

print("\nOne-Way ANOVA")
print("F-statistic:", anova_statistic)
print("P-value:", anova_p_value)

if anova_p_value < 0.05:
    print(
        "At least one BMI group has a statistically "
        "different mean glucose level."
    )
else:
    print(
        "No statistically significant difference "
        "between the BMI groups."
    )

df_clean["Age_Group"] = pd.cut(
    df_clean["Age"],
    bins=[0, 30, 50, 100],
    labels=[
        "Young",
        "Middle",
        "Older"
    ]
)

print("\nAge groups:")
print(df_clean["Age_Group"].value_counts())

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df_clean["BMI_Category"] = df_clean["BMI"].apply(
    classify_bmi
)

print("\nBMI categories:")
print(df_clean["BMI_Category"].value_counts())

X = df_clean.drop(
    columns=[
        "Outcome",
        "Age_Group",
        "BMI_Category",
        "High_Glucose"
    ]
)

y = df_clean["Outcome"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled feature data:")
print(X_scaled_df.head())

selector = SelectKBest(
    score_func=f_classif,
    k=5
)

X_selected = selector.fit_transform(
    X_scaled,
    y
)

selected_features = X.columns[
    selector.get_support()
]

print("\nSelected features:")
print(selected_features.tolist())

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "F_Score": selector.scores_,
    "P_Value": selector.pvalues_
})

feature_scores = feature_scores.sort_values(
    by="F_Score",
    ascending=False
)

print("\nFeature scores:")
print(feature_scores)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Outcome": y
})

print("\nPCA transformed data:")
print(pca_df.head())

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Outcome"
)

plt.title("PCA Visualization")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TABULAR_DIR,
        "pca_visualization.png"
    )
)

plt.show()

cleaned_file = os.path.join(
    TABULAR_DIR,
    "cleaned_diabetes.csv"
)

df_clean.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned dataset saved at:")
print(cleaned_file)

print("\nFinal shape:")
print(df_clean.shape)

print("\nFinal missing values:")
print(df_clean.isnull().sum())

print("\nFinal duplicate count:")
print(df_clean.duplicated().sum())

print("\nFinal columns:")
print(df_clean.columns.tolist())

print("\nPART 1 - TABULAR DATA COMPLETED")


#TEXTUAL DATA---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


TEXTUAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "textual"
)

TEXTUAL_FILE = os.path.join(
    TEXTUAL_DIR,
    "mtsamples.csv"
)


df = pd.read_csv(TEXTUAL_FILE)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


text_column = None

possible_columns = [
    "transcription",
    "Transcription",
    "text",
    "Text",
    "medical_transcription",
    "Medical Transcription"
]

for column in possible_columns:
    if column in df.columns:
        text_column = column
        break

if text_column is None:
    for column in df.columns:
        if df[column].dtype == "object":
            sample_text = df[column].dropna().astype(str)

            if len(sample_text) > 0:
                average_length = sample_text.str.len().mean()

                if average_length > 100:
                    text_column = column
                    break

if text_column is None:
    raise ValueError(
        "Could not identify the medical transcription column."
    )

print("\nText column identified:")
print(text_column)


print("\nSample medical transcription:")
print(df[text_column].dropna().iloc[0])


def clean_text(text):

    text = str(text)

    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]',
        text
    )

    text = re.sub(
        r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b',
        '[PHONE]',
        text
    )

    text = re.sub(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        '[DATE]',
        text
    )

    text = re.sub(
        r'\b\d{5}(?:-\d{4})?\b',
        '[ZIP]',
        text
    )

    text = text.lower()

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


df["cleaned_text"] = df[text_column].fillna("").apply(
    clean_text
)


print("\nCleaned medical transcription:")
print(df["cleaned_text"].iloc[0])


def tokenize(text):

    tokens = re.findall(
        r'\b[a-z]+\b',
        text
    )

    return tokens


df["tokens"] = df["cleaned_text"].apply(
    tokenize
)


print("\nTokens from first transcription:")
print(df["tokens"].iloc[0][:50])


word_frequency = {}

for tokens in df["tokens"]:

    for word in tokens:

        if word in word_frequency:
            word_frequency[word] += 1

        else:
            word_frequency[word] = 1


word_frequency_df = pd.DataFrame(
    word_frequency.items(),
    columns=[
        "Word",
        "Frequency"
    ]
)

word_frequency_df = word_frequency_df.sort_values(
    by="Frequency",
    ascending=False
)

print("\nTop 20 most frequent words:")

print(
    word_frequency_df.head(20)
)


top_words = word_frequency_df.head(20)

plt.figure(figsize=(12, 7))

plt.bar(
    top_words["Word"],
    top_words["Frequency"]
)

plt.title("Top 20 Most Frequent Words in Medical Transcriptions")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        TEXTUAL_DIR,
        "top_20_word_frequency.png"
    ),
    dpi=300
)

plt.show()


df["word_count"] = df["tokens"].apply(
    len
)

df["character_count"] = df["cleaned_text"].apply(
    len
)


print("\nText feature statistics:")

print(
    df[
        [
            "word_count",
            "character_count"
        ]
    ].describe()
)


plt.figure(figsize=(8, 5))

plt.hist(
    df["word_count"],
    bins=30
)

plt.title("Distribution of Medical Transcription Word Counts")
plt.xlabel("Number of Words")
plt.ylabel("Number of Transcriptions")

plt.tight_layout()

plt.savefig(
    os.path.join(
        TEXTUAL_DIR,
        "word_count_distribution.png"
    ),
    dpi=300
)

plt.show()


df["token_sequence"] = df["tokens"].apply(
    lambda x: " ".join(x)
)


cleaned_file = os.path.join(
    TEXTUAL_DIR,
    "cleaned_mtsamples.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)


word_frequency_file = os.path.join(
    TEXTUAL_DIR,
    "word_frequency.csv"
)

word_frequency_df.to_csv(
    word_frequency_file,
    index=False
)


print("\nCleaned dataset saved at:")
print(cleaned_file)

print("\nWord frequency data saved at:")
print(word_frequency_file)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nPART 2 - TEXTUAL DATA COMPLETED")

#IMAGE DATA----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "image"
)

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)

image_files = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_files.append(
                os.path.join(root, file)
            )

print("\nTotal images found:")
print(len(image_files))

if len(image_files) == 0:
    raise ValueError("No image files found in the image folder.")

print("\nSample image paths:")
for path in image_files[:5]:
    print(path)

sample_image_path = image_files[0]

sample_image = Image.open(
    sample_image_path
)

print("\nSample image information:")
print("Format:", sample_image.format)
print("Size:", sample_image.size)
print("Mode:", sample_image.mode)

plt.figure(figsize=(8, 6))

plt.imshow(
    sample_image,
    cmap="gray"
)

plt.title("Sample Chest X-Ray Image")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "sample_chest_xray.png"
    ),
    dpi=300
)

plt.show()

class_counts = {}

for path in image_files:

    folder_name = os.path.basename(
        os.path.dirname(path)
    )

    if folder_name in class_counts:
        class_counts[folder_name] += 1
    else:
        class_counts[folder_name] = 1

print("\nClass distribution:")

for class_name, count in class_counts.items():
    print(class_name, ":", count)

class_names = list(class_counts.keys())
class_values = list(class_counts.values())

plt.figure(figsize=(8, 5))

plt.bar(
    class_names,
    class_values
)

plt.title("Chest X-Ray Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "class_distribution.png"
    ),
    dpi=300
)

plt.show()

processed_images = []
image_features = []

sample_paths = image_files[:100]

for image_path in sample_paths:

    image = Image.open(
        image_path
    )

    image = image.convert("L")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = cv2.medianBlur(
        image_array,
        3
    )

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    processed_images.append(
        image_array
    )

    image_features.append(
        [
            image_array.mean(),
            image_array.std(),
            image_array.min(),
            image_array.max()
        ]
    )

processed_images = np.array(
    processed_images
)

image_features_df = pd.DataFrame(
    image_features,
    columns=[
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity"
    ]
)

print("\nProcessed image tensor shape:")
print(processed_images.shape)

print("\nImage feature statistics:")
print(image_features_df.describe())

processed_sample = processed_images[0].squeeze()

plt.figure(figsize=(8, 6))

plt.imshow(
    processed_sample,
    cmap="gray"
)

plt.title("Processed Chest X-Ray Image")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "processed_chest_xray.png"
    ),
    dpi=300
)

plt.show()

augmented_image = processed_sample.copy()

angle = np.random.uniform(
    -5,
    5
)

height, width = augmented_image.shape

center = (
    width // 2,
    height // 2
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    angle,
    1.0
)

augmented_image = cv2.warpAffine(
    augmented_image,
    rotation_matrix,
    (width, height)
)

contrast_factor = np.random.uniform(
    0.9,
    1.1
)

augmented_image = np.clip(
    augmented_image * contrast_factor,
    0,
    1
)

plt.figure(figsize=(8, 6))

plt.imshow(
    augmented_image,
    cmap="gray"
)

plt.title("Augmented Chest X-Ray Image")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "augmented_chest_xray.png"
    ),
    dpi=300
)

plt.show()

processed_sample_uint8 = (
    processed_sample * 255
).astype(np.uint8)

anonymized_image = Image.fromarray(
    processed_sample_uint8
)

anonymized_image.save(
    os.path.join(
        IMAGE_DIR,
        "anonymized_chest_xray.png"
    ),
    format="PNG"
)

image_features_file = os.path.join(
    IMAGE_DIR,
    "image_features.csv"
)

image_features_df.to_csv(
    image_features_file,
    index=False
)

print("\nImage feature data saved at:")
print(image_features_file)

print("\nProcessed images:")
print(len(processed_images))

print("\nImage tensor dimensions:")
print(processed_images.shape)

print("\nPART 3 - IMAGE DATA COMPLETED")

#SIGNAL------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SIGNAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "signal",
    "mit-bih-arrhythmia-database-1.0.0"
)

record_name = "100"

record_path = os.path.join(
    SIGNAL_DIR,
    record_name
)

record = wfdb.rdrecord(
    record_path
)

annotation = wfdb.rdann(
    record_path,
    "atr"
)

print("\nECG record loaded successfully!")

print("\nSignal information:")
print("Sampling frequency:", record.fs)
print("Number of samples:", record.sig_len)
print("Number of signals:", record.n_sig)
print("Signal names:", record.sig_name)
print("Units:", record.units)

print("\nAnnotation information:")
print("Number of annotations:", len(annotation.sample))
print("Annotation symbols:", annotation.symbol[:30])

ecg_signal = record.p_signal[:, 0]

print("\nECG signal statistics:")
print("Mean:", np.mean(ecg_signal))
print("Standard deviation:", np.std(ecg_signal))
print("Minimum:", np.min(ecg_signal))
print("Maximum:", np.max(ecg_signal))

samples_to_plot = min(
    2000,
    len(ecg_signal)
)

time = np.arange(
    samples_to_plot
) / record.fs

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    ecg_signal[:samples_to_plot]
)

plt.title("MIT-BIH ECG Record 100")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SIGNAL_DIR,
        "ecg_record_100.png"
    ),
    dpi=300
)

plt.show()

b, a = scipy.signal.butter(
    3,
    [0.5, 40],
    btype="bandpass",
    fs=record.fs
)

filtered_ecg = scipy.signal.filtfilt(
    b,
    a,
    ecg_signal
)

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    filtered_ecg[:samples_to_plot]
)

plt.title("Filtered ECG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SIGNAL_DIR,
        "filtered_ecg_record_100.png"
    ),
    dpi=300
)

plt.show()

window_duration = 2

window_size = int(
    window_duration * record.fs
)

windows = []

features = []

for start in range(
    0,
    len(filtered_ecg) - window_size + 1,
    window_size
):

    window = filtered_ecg[
        start:start + window_size
    ]

    windows.append(
        window
    )

    features.append(
        [
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
            np.median(window)
        ]
    )

windows = np.array(
    windows
)

ecg_features_df = pd.DataFrame(
    features,
    columns=[
        "mean",
        "std",
        "min",
        "max",
        "median"
    ]
)

print("\nECG window information:")
print("Window duration:", window_duration, "seconds")
print("Window size:", window_size, "samples")
print("Number of windows:", len(windows))
print("Window shape:", windows.shape)

print("\nECG feature statistics:")
print(
    ecg_features_df.describe()
)

plt.figure(figsize=(14, 5))

plt.plot(
    windows[0]
)

plt.title("First 2-Second ECG Window")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SIGNAL_DIR,
        "first_ecg_window.png"
    ),
    dpi=300
)

plt.show()

ecg_features_file = os.path.join(
    SIGNAL_DIR,
    "ecg_features.csv"
)

ecg_features_df.to_csv(
    ecg_features_file,
    index=False
)

print("\nECG feature data saved at:")
print(ecg_features_file)

print("\nFinal ECG feature shape:")
print(ecg_features_df.shape)

print("\nPART 4 - SIGNAL DATA COMPLETED")