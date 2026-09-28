import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from query_features import (
    extract_query_features,
    ENGINEERED_FEATURES
)
DATASET_PATH = "router_dataset.csv"
MODEL_OUTPUT = "router_model.pkl"
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
RANDOM_STATE = 42
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)
df = pd.read_csv(DATASET_PATH)
print("Dataset shape:", df.shape)
df = df.dropna(
    subset=[
        "query",
        "best_pipeline"
    ]
).copy()
df["best_pipeline"] = (
    df["best_pipeline"]
    .astype(str)
    .str.strip()
    .str.lower()
)
df["best_pipeline"] = (
    df["best_pipeline"]
    .replace({
        "parent": "parent_document"
    })
)
print("\nPipeline distribution:")
print(
    df["best_pipeline"]
    .value_counts()
)
class_counts = (
    df["best_pipeline"]
    .value_counts()
)
if (class_counts < 2).any():
    print("\nERROR:")
    print(
        "Every pipeline needs at least "
        "2 examples for stratified splitting."
    )
    print(class_counts)
    raise ValueError(
        "Insufficient samples in one or more classes."
    )
queries = df["query"].astype(str)
y = df["best_pipeline"]
X_train_text, X_test_text, y_train, y_test = (
    train_test_split(
        queries,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)
print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))
print("\n" + "=" * 70)
print("LOADING SENTENCE TRANSFORMER")
print("=" * 70)
embedder = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)
print("\nGenerating training embeddings...")
train_embeddings = embedder.encode(
    X_train_text.tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)
print("\nGenerating testing embeddings...")
test_embeddings = embedder.encode(
    X_test_text.tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)
print(
    "\nEmbedding shape:",
    train_embeddings.shape
)
print("\n" + "=" * 70)
print("EXTRACTING ENGINEERED FEATURES")
print("=" * 70)
train_features = pd.DataFrame(
    [
        extract_query_features(query)
        for query in X_train_text
    ]
)
test_features = pd.DataFrame(
    [
        extract_query_features(query)
        for query in X_test_text
    ]
)
train_features = train_features[
    ENGINEERED_FEATURES
]
test_features = test_features[
    ENGINEERED_FEATURES
]
print(
    "\nEngineered feature shape:",
    train_features.shape
)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(
    train_features
)
test_features_scaled = scaler.transform(
    test_features
)
X_train = np.hstack(
    [
        train_embeddings,
        train_features_scaled
    ]
)
X_test = np.hstack(
    [
        test_embeddings,
        test_features_scaled
    ]
)
print(
    "\nFinal training feature shape:",
    X_train.shape
)
print(
    "Final testing feature shape:",
    X_test.shape
)
print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
model.fit(
    X_train,
    y_train
)
predictions = model.predict(
    X_test
)
accuracy = accuracy_score(
    y_test,
    predictions
)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print(
    f"\nAccuracy: {accuracy:.4f}"
)
print(
    f"Macro F1: {macro_f1:.4f}"
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)
artifact = {
    "model":
        model,
    "scaler":
        scaler,
    "embedding_model":
        EMBEDDING_MODEL_NAME,
    "engineered_features":
        ENGINEERED_FEATURES,
    "version":
        "v2",
    "accuracy":
        accuracy,
    "macro_f1":
        macro_f1
}
joblib.dump(
    artifact,
    MODEL_OUTPUT
)
print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)
print(
    f"\nSaved to: {MODEL_OUTPUT}"
)
