import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from query_features import extract_query_features
MODEL_PATH = "router_model.pkl"
artifact = joblib.load(
    MODEL_PATH
)
model = artifact["model"]
scaler = artifact["scaler"]
embedding_model_name = (
    artifact["embedding_model"]
)
engineered_features = (
    artifact["engineered_features"]
)
embedder = SentenceTransformer(
    embedding_model_name
)
def route_query(query: str):
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    features = extract_query_features(
        query
    )
    features_df = pd.DataFrame(
        [features]
    )
    features_df = features_df[
        engineered_features
    ]
    features_scaled = scaler.transform(
        features_df
    )
    X = np.hstack(
        [
            embedding,
            features_scaled
        ]
    )
    prediction = model.predict(
        X
    )[0]
    probabilities = None
    confidence = None
    if hasattr(
        model,
        "predict_proba"
    ):
        probs = model.predict_proba(
            X
        )[0]
        confidence = float(
            np.max(probs)
        )
        probabilities = {
            label: float(prob)
            for label, prob
            in zip(
                model.classes_,
                probs
            )
        }
    return {
        "pipeline":
            prediction,
        "confidence":
            confidence,
        "probabilities":
            probabilities
    }
