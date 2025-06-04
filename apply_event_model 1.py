# event_model_utils.py
import os
import pickle
import pandas as pd
from train_event_model import train_event_model

MODEL_PATH = 'event_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

def ensure_model_trained():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(ENCODER_PATH)):
        train_event_model()  # Will default to using "Large_Sample Event_Mapping.xlsx"

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    return clf, tfidf, label_encoder

def predict_and_add_events(file_path):
    ensure_model_trained()

    clf, tfidf, label_encoder = load_model()

    df = pd.read_csv(file_path)
    for col in ['event_category', 'event_label', 'event_action']:
        df[col] = df[col].fillna('').astype(str)
    df['input_sequence'] = df[['event_category', 'event_label', 'event_action']].agg(' '.join, axis=1)

    X = tfidf.transform(df['input_sequence'])
    preds = clf.predict(X)
    df['event'] = label_encoder.inverse_transform(preds)

    return df
