import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("FraudShield AI - Model Training")
print("=" * 50)

# Load Dataset
df = pd.read_csv("datasets/creditcard.csv")

print("\nDataset Loaded Successfully")
print("Rows :", len(df))
print("Columns :", len(df.columns))

# Select Features
X = df[["Time", "Amount"]]
y = df["Class"]

print("\nTraining Features:")
print(X.columns.tolist())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Create Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")
model.fit(X_train, y_train)

print("Training Completed Successfully")

# Prediction
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# Create Folder
os.makedirs("trained_models", exist_ok=True)

# Save Model
model_path = "trained_models/fraud_model.pkl"

joblib.dump(model, model_path)

print("\nModel Saved Successfully")
print("Location :", model_path)

print("\nFraudShield AI Model Ready")