
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load OCR data
df = pd.read_csv("dataset/receipt_ocr.csv")
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

# Assign initial labels using OCR keywords
def assign_label(text):
    text = text.lower()

    if "utility bill" in text or "electricity bill" in text:
        return "Bill"

    if "tax invoice" in text or "sales invoice" in text:
        return "Invoice"

    if "cash bill" in text:
        return "Bill"

    if "receipt" in text or "thank you" in text:
        return "Receipt"

    return "Receipt"


df["label"] = df["text"].apply(assign_label)

print("\nAutomatically assigned labels:")
print(df["label"].value_counts())

df = df[df["text"].str.strip() != ""]

print("Total documents:", len(df))
print("\nLabels:")
print(df["label"].value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])
print("Unique classes:", y_train.nunique())
print("Class names:", y_train.unique())

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nTest accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/document_classifier.pkl")

print("\nModel saved at models/document_classifier.pkl")
