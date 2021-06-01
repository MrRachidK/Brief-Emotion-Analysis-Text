# Import of all the libraries we need to work

import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text")
import pandas as pd
import numpy as np
import xgboost
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,auc, roc_curve, roc_auc_score,accuracy_score
import matplotlib.pyplot as plt
from src.functions import *
import joblib
filename_xgboost = 'xg_boost.sav'


#### Etape 1

# Import of our databases
emotion_final, text_emotion = import_database()

# Database 1

# Encoding of the emotion column
label_encoding(emotion_final, [(emotion_final['label'] == "sadness") | (emotion_final['label'] == "fear") | (emotion_final['label'] == "anger"), (emotion_final['label'] == "love") | (emotion_final['label'] == "surprise") | (emotion_final['label'] == "happy")] , [0, 1])

# Division of our data by using train_test_split
X_train5, X_test5, y_train5, y_test5 = variables_selection(emotion_final['text'], emotion_final['binary_emotion'])

# Vectorization of our data
vectorizer5 = vectorize_data(CountVectorizer, emotion_final['text'], None)

# Transformation of our data in order to use our model on it
text_train5, text_test5 = transform_training_data(vectorizer5, X_train5, X_test5)

# Building of our data
model5 = define_model(xgboost.XGBClassifier(), text_train5, y_train5)

# Predictions of target values for the testing data
y_pred5, y_proba5, dataframe5 = predict_model(model5, text_test5, X_test5)

# Calculation of the accuracy
accuracy5 = calculate_score(model5, text_test5, y_test5)


# Database 2

# Encoding of the emotion column
label_encoding(text_emotion, [(text_emotion['label'] == "empty") | (text_emotion['label'] == "sadness") | (text_emotion['label'] == "worry") | (text_emotion['label'] == "hate") | (text_emotion['label'] == "boredom") | (text_emotion['label'] == "anger"), (text_emotion['label'] == "enthusiasm") | (text_emotion['label'] == "neutral") | (text_emotion['label'] == "surprise") | (text_emotion['label'] == "love") | (text_emotion['label'] == "fun") | (text_emotion['label'] == "happiness") | (text_emotion['label'] == "relief")] , [0, 1])

# Division of our data by using train_test_split
X_train6, X_test6, y_train6, y_test6 = variables_selection(text_emotion['text'], text_emotion['binary_emotion'])

# Vectorization of our data
vectorizer6 = vectorize_data(CountVectorizer, text_emotion['text'], None)

# Transformation of our data in order to use our model on it
text_train6, text_test6 = transform_training_data(vectorizer6, X_train6, X_test6)

# Building of our data
model6 = define_model(xgboost.XGBClassifier(), text_train6, y_train6)

# Predictions of target values for the testing data
y_pred6, y_proba6, dataframe6 = predict_model(model6, text_test6, X_test6)

# Calculation of the accuracy
accuracy6 = calculate_score(model6, text_test6, y_test6)


#### Etape 2

# Concatenation of the two databases
new_emotion_text = concat_databases(emotion_final, text_emotion)
new_emotion_text["label"].replace({'happiness': 'happy'}, inplace=True)

# Encoding of the emotion column
label_encoding(new_emotion_text, [(new_emotion_text['label'] == "empty") | (new_emotion_text['label'] == "sadness") | (new_emotion_text['label'] == "worry") | (new_emotion_text['label'] == "hate") | (new_emotion_text['label'] == "boredom") | (new_emotion_text['label'] == "anger") | new_emotion_text['label'] == "fear", (new_emotion_text['label'] == "enthusiasm") | (new_emotion_text['label'] == "neutral") | (new_emotion_text['label'] == "surprise") | (new_emotion_text['label'] == "love") | (new_emotion_text['label'] == "fun") | (new_emotion_text['label'] == "happy") | (new_emotion_text['label'] == "relief")], [0, 1])

# Division of our data by using train_test_split
X_train7, X_test7, y_train7, y_test7 = variables_selection(new_emotion_text['text'], new_emotion_text['binary_emotion'])

# Vectorization of our data
vectorizer7 = vectorize_data(CountVectorizer, new_emotion_text['text'], None)

# Transformation of our data in order to use our model on it
text_train7, text_test7 = transform_training_data(vectorizer7, X_train7, X_test7)

# Building of our data
model7 = define_model(xgboost.XGBClassifier(), text_train7, y_train7)

# Predictions of target values for the testing data
y_pred7, y_proba7, dataframe7 = predict_model(model7, text_test7, X_test7)

# Calculation of the accuracy
accuracy7 = calculate_score(model7, text_test7, y_test7)

#### Etape 3

# Selection of the trained datas and the tested datas
X_svm, y_svm, X_test8, y_test8 = step3_variables(emotion_final, text_emotion)

# Vectorization of our data
vectorizer8 = vectorize_data(CountVectorizer, X_svm, X_test8, 1)

# Transformation of our data in order to use our model on it
text_train8, text_test8 = transform_training_data(vectorizer8, X_svm, X_test8)

# Building of our data
model8 = define_model(xgboost.XGBClassifier(), text_train8, y_svm)

# Predictions of target values for the testing data
y_pred8, y_proba8, dataframe8 = predict_model(model8, text_test8, X_test8)

# Calculation of the part of correctly predicted results
accuracy8 = calculate_accuracy_score(y_test8, y_pred8)

# Saving of the useful objects for our Streamlit app
joblib.dump([dataframe5, accuracy5, dataframe6, accuracy6, dataframe7, accuracy7, dataframe8, accuracy8, y_pred5, y_pred6, y_pred7, y_proba5, y_proba6, y_proba7, y_test5, y_test6, y_test7], filename_xgboost)