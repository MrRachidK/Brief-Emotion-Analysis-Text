# Import of all the libraries we need to work

import sys
from altair.vegalite.v4.api import value
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from src.functions import *
import joblib
filename_randomforest = "random_forest.sav"

#### Etape 1

# Import of our databases
emotion_final, text_emotion = import_database()

# Database 1

# Encoding of the emotion column
label_encoding(emotion_final, [(emotion_final['label'] == "sadness") | (emotion_final['label'] == "fear") | (emotion_final['label'] == "anger"), (emotion_final['label'] == "love") | (emotion_final['label'] == "surprise") | (emotion_final['label'] == "happy")], [0, 1])

# Division of our data by using train_test_split
X_train9, X_test9, y_train9, y_test9 = variables_selection(emotion_final['text'], emotion_final['binary_emotion'])

# Vectorization of our data
vectorizer9 = vectorize_data(CountVectorizer, emotion_final['text'], None)

# Transformation of our data in order to use our model on it
text_train9, text_test9 = transform_training_data(vectorizer9, X_train9, X_test9)

# Building of our model
model9 = define_model(RandomForestClassifier(random_state=42), text_train9, y_train9)

# Predictions of target values for the testing data
y_pred9, y_proba9, dataframe9 = predict_model(model9, text_test9, X_test9)

# Calculation of the train scores
train_score1 = calculate_score(model9, text_train9, y_train9)
train_score2 = calculate_score(model9, text_test9, y_test9)

# Database 2

# Encoding of the emotion column
label_encoding(text_emotion, [(text_emotion['label'] == "empty") | (text_emotion['label'] == "sadness") | (text_emotion['label'] == "worry") | (text_emotion['label'] == "hate") | (text_emotion['label'] == "boredom") | (text_emotion['label'] == "anger"), (text_emotion['label'] == "enthusiasm") | (text_emotion['label'] == "neutral") | (text_emotion['label'] == "surprise") | (text_emotion['label'] == "love") | (text_emotion['label'] == "fun") | (text_emotion['label'] == "happiness") | (text_emotion['label'] == "relief")] ,[0, 1])

# Division of our data by using train_test_split
X_train10, X_test10, y_train10, y_test10 = variables_selection(text_emotion['text'], text_emotion['binary_emotion'])

# Vectorization of our data
vectorizer10 = vectorize_data(CountVectorizer, text_emotion['text'], None)

# Transformation of our data in order to use our model on it
text_train10, text_test10 = transform_training_data(vectorizer10, X_train10, X_test10)

# Building of our model
model10 = define_model(RandomForestClassifier(random_state=42), text_train10, y_train10)

# Predictions of target values for the testing data
y_pred10, y_proba10, dataframe10 = predict_model(model10, text_test10, X_test10)

# Calculation of the train scores
train_score3 = calculate_score(model10, text_train10, y_train10)
train_score4 = calculate_score(model10, text_test10, y_test10)

#### Etape 2

# Concatenation of the two databases
new_emotion_text = concat_databases(emotion_final, text_emotion)
new_emotion_text["label"].replace({'happiness': 'happy'}, inplace=True)

# Encoding of the emotion column
label_encoding(new_emotion_text, [(new_emotion_text['label'] == "empty") | (new_emotion_text['label'] == "sadness") | (new_emotion_text['label'] == "worry") | (new_emotion_text['label'] == "hate") | (new_emotion_text['label'] == "boredom") | (new_emotion_text['label'] == "anger") | new_emotion_text['label'] == "fear", (new_emotion_text['label'] == "enthusiasm") | (new_emotion_text['label'] == "neutral") | (new_emotion_text['label'] == "surprise") | (new_emotion_text['label'] == "love") | (new_emotion_text['label'] == "fun") | (new_emotion_text['label'] == "happy") | (new_emotion_text['label'] == "relief")] ,[0, 1])

# Division of our data by using train_test_split
X_train11, X_test11, y_train11, y_test11 = variables_selection(new_emotion_text['text'], new_emotion_text['binary_emotion'])

# Vectorization of our data
vectorizer11 = vectorize_data(CountVectorizer, new_emotion_text['text'], None)

# Transformation of our data in order to use our model on it
text_train11, text_test11 = transform_training_data(vectorizer11, X_train11, X_test11)

# Building of our model
model11 = define_model(RandomForestClassifier(random_state=42), text_train11, y_train11)

# Predictions of target values for the testing data
y_pred11, y_proba11, dataframe11 = predict_model(model11, text_test11, X_test11)

# Calculation of the train scores
train_score5 = calculate_score(model11, text_train11, y_train11)
train_score6 = calculate_score(model11, text_test11, y_test11)

#### Etape 3

# Selection of the trained datas and the tested datas
X_rf, y_rf, X_test12, y_test12 = step3_variables(emotion_final, text_emotion)

# Vectorization of our data
vectorizer12 = vectorize_data(CountVectorizer, X_rf, X_test12, 1)

# Transformation of our data in order to use our model on it
text_train12, text_test12 = transform_training_data(vectorizer12, X_rf, X_test12)

# Building of our model
model12 = define_model(RandomForestClassifier(random_state=42), text_train12, y_rf)

# Predictions of target values for the testing data
y_pred12, y_proba12, dataframe12 = predict_model(model12, text_test12, X_test12)

# Calculation of the part of correctly predicted results
accuracy9 = calculate_accuracy_score(y_test12, y_pred12)

# Saving of the useful objects for our Streamlit app
joblib.dump([dataframe9, dataframe10, dataframe11, dataframe12, train_score1, train_score2, train_score3, train_score4, train_score5, train_score6, accuracy9], filename_randomforest)
