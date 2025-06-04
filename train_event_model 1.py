# train_event_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import os

MODEL_PATH = 'event_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

def train_event_model(data_path="Large_Sample Event_Mapping.xlsx"):
    print("Training new model...")

    df = pd.read_excel(data_path)

    for col in ['event_category', 'event_label', 'event_action']:
        df[col] = df[col].fillna('').astype(str)
    df['input_sequence'] = df[['event_category', 'event_label', 'event_action']].agg(' '.join, axis=1)

    class_counts = df['Event'].value_counts()
    df = df[df['Event'].isin(class_counts[class_counts >= 2].index)]

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Event'])

    X_train, _, y_train, _ = train_test_split(
        df['input_sequence'], df['label'],
        test_size=0.3,
        random_state=42,
        stratify=df['label']
    )

    unique, counts = np.unique(y_train, return_counts=True)
    labels_to_keep = unique[counts >= 2]
    train_mask = np.isin(y_train, labels_to_keep)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)

    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_resampled, y_resampled)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Model training complete and saved.")
