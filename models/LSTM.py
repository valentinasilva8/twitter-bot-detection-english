# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:35:20 2025

"""

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from textblob import TextBlob

'''LSTM Model'''
# Prepare the hybrid features
def prepare_hybrid_features(df, max_words=15000, max_len=100):

    # Tokenizer for text features
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    texts = [' '.join(tweet) if isinstance(tweet, list) else str(tweet) for tweet in df['processed_tweet']]
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')


    # Numerical features
    numerical_features = [
        'num_links', 'num_hashtags', 'num_mentions', 'num_chars',
        'num_followers', 'num_friends', 'is_verified', 'account_age',
        'tweet_length', 'statuses_count', 'favorites_count', 'listed_count',
        'friends_follower_ratio'
    ]
    numerical_data = df[numerical_features].values

    return padded_sequences, numerical_data, tokenizer


# Create the hybrid LSTM model
def create_hybrid_lstm_model(max_words, max_len, num_numerical_features, embedding_dim=100, lstm_units=32):
    # Text input branch
    text_input = Input(shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(text_input)
    lstm_out = LSTM(lstm_units, return_sequences=True)(embedding)
    lstm_out = LSTM(lstm_units)(lstm_out)
    lstm_out = Dropout(0.7)(lstm_out)  # Increased dropout rate

    # Numerical input branch
    numerical_input = Input(shape=(num_numerical_features,))
    numerical_dense = Dense(32, activation='relu')(numerical_input)
    numerical_dense = Dropout(0.3)(numerical_dense)

    # Combine both branches
    combined = concatenate([lstm_out, numerical_dense])

    # Output layers
    dense = Dense(64, activation='relu')(combined)
    dense = Dropout(0.6)(dense)  # Increased dropout rate
    output = Dense(1, activation='sigmoid')(dense)

    # Create model
    hybrid_lstm_model = Model(inputs=[text_input, numerical_input], outputs=output)

    # Compile model
    hybrid_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    return hybrid_lstm_model



# Train the hybrid LSTM model
def train_hybrid_lstm_model(df, max_words=15000, max_len=100, epochs=10, batch_size=32):
    # Prepare features
    padded_sequences, numerical_data, tokenizer = prepare_hybrid_features(df, max_words, max_len)
    labels = df['label'].values

    # Apply ADASYN for oversampling
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(np.hstack((padded_sequences, numerical_data)), labels)

    # Split resampled data into text and numerical parts
    X_text = X_resampled[:, :max_len]
    X_num = X_resampled[:, max_len:]

    # Split the data into training and testing sets
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    # Create model
    hybrid_lstm_model = create_hybrid_lstm_model(
        max_words=max_words,
        max_len=max_len,
        num_numerical_features=numerical_data.shape[1]
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Reduce patience to prevent overfitting
    model_checkpoint = ModelCheckpoint('best_hybrid_lstm_model.keras', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)  # Added learning rate scheduler

    # Train model
    hybrid_lstm_model.fit(
        [X_text_train, X_num_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.3,
       callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    y_pred_proba = hybrid_lstm_model.predict([X_text_test, X_num_test])
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    return hybrid_lstm_model,[X_text_test, X_num_test], y_test, y_pred



'''Bidirectional LSTM with different parameters and sentiment features'''
# Prepare Features
def prepare_bd_lstm_features(df, max_words=20000, max_len=100):

    # Prepare text features
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    texts = [' '.join(tweet) if isinstance(tweet, list) else str(tweet) for tweet in df['processed_tweet']]
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Prepare numerical features
     # Extract sentiment features

    df['sentiment_subjectivity'] = df['processed_tweet'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    numerical_features = [
        'account_age', 'favorites_count', 'friends_follower_ratio',
        'listed_count', 'num_followers', 'num_friends', 'tweet_length',
        'statuses_count', 'is_verified', 'num_links', 'num_hashtags',
        'num_mentions', 'num_chars', 'sentiment_subjectivity']
    numerical_data = df[numerical_features].values

    return padded_sequences, numerical_data, tokenizer

# Create BI-LSTM Model that combines text and numerical features
def create_bd_lstm_model(max_words, max_len, num_numerical_features, embedding_dim=100, lstm_units=32):

    # Text input branch
    text_input = Input(shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(text_input)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding)
    lstm_out = Bidirectional(LSTM(lstm_units))(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)  # Increase dropout to reduce overfitting

    # Numerical input branch
    numerical_input = Input(shape=(num_numerical_features,))
    numerical_dense = Dense(32, activation='relu')(numerical_input)
    numerical_dense = Dropout(0.3)(numerical_dense)

    # Combine both branches
    combined = concatenate(([lstm_out, numerical_dense]))

    # Output layers
    dense = Dense(64, activation='relu')(combined)
    dense = Dropout(0.4)(dense)  # Increase dropout to reduce overfitting
    output = Dense(1, activation='sigmoid')(dense)

    # Create model
    bd_lstm_model = Model(inputs=[text_input, numerical_input], outputs=output)

    # Compile model
    bd_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    return bd_lstm_model



def train_bd_lstm_model(df, max_words=20000, max_len=100, epochs=10, batch_size=32):

    # Prepare features
    padded_sequences, numerical_data, tokenizer = prepare_bd_lstm_features(df, max_words, max_len)

    # Prepare labels
    labels = df['label'].values

    # Apply ADASYN for class balancing / Can also Try SMOTE or SMOTEENN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(np.hstack((padded_sequences, numerical_data)), labels)

    # Split the data
    X_text = X_resampled[:, :max_len]
    X_num = X_resampled[:, max_len:]
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    # Create and train model
    bd_lstm_model = create_bd_lstm_model(
        max_words=max_words,
        max_len=max_len,
        num_numerical_features=X_num.shape[1]
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_accuracy', save_best_only=True)

    # Train model
    bd_lstm_model.fit(
        [X_text_train, X_num_train],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.3,
        callbacks=[early_stopping, model_checkpoint]
    )

    y_pred_proba = bd_lstm_model.predict([X_text_test, X_num_test])
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    return bd_lstm_model,[X_text_test, X_num_test], y_test, y_pred



