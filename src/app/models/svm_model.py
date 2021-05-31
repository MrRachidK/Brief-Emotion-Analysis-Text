# Import of all the libraries we need to work

import sys
from altair.vegalite.v4.api import value
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.functions import plot_roc
from src.text_preprocessing import new_list, new_list2
from src.functions import *
import joblib
filename_svm = "svm_model.sav"

#### Etape 1

emotion_final, text_emotion = import_database()

# Database 1

label_encoding_trinary(emotion_final, [(emotion_final['label'] == "sadness") | (emotion_final['label'] == "anger"), (emotion_final['label'] == "fear") | (emotion_final['label'] == "surprise"), (emotion_final['label'] == "love") | (emotion_final['label'] == "happy")], [0, 1, 2])

X_train13, X_test13, y_train13, y_test13 = variables_selection(emotion_final['text'], emotion_final['trinary_emotion'])
vectorizer13 = vectorize_tfidf_data(TfidfVectorizer, emotion_final['text'], None)
text_train13, text_test13 = transform_training_data(vectorizer13, X_train13, X_test13)
model13 = define_model(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), text_train13, y_train13)
y_pred13, dataframe13 = predict_model(model13, text_test13, X_test13)
accuracy10 = calculate_accuracy_score(y_test13, y_pred13)


# Database 2

label_encoding_trinary(text_emotion, [(text_emotion['label'] == "sadness") | (text_emotion['label'] == "worry") | (text_emotion['label'] == "anger") | (text_emotion['label'] == "hate"), (text_emotion['label'] == "neutral")| (text_emotion['label'] == "boredom") | (text_emotion['label'] == "surprise") | (text_emotion['label'] == "empty"), (text_emotion['label'] == "love") | (text_emotion['label'] == "fun") | (text_emotion['label'] == "happiness") | (text_emotion['label'] == "relief") | (text_emotion['label'] == "enthusiasm")], [0, 1, 2])

X_train14, X_test14, y_train14, y_test14 = variables_selection(text_emotion['text'], text_emotion['trinary_emotion'])
vectorizer14 = vectorize_tfidf_data(TfidfVectorizer, text_emotion['text'], None)
text_train14, text_test14 = transform_training_data(vectorizer14, X_train14, X_test14)
model14 = define_model(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), text_train14, y_train14)
y_pred14, dataframe14 = predict_model(model14, text_test14, X_test14)
accuracy11 = calculate_accuracy_score(y_test14, y_pred14)


#### Etape 2

new_emotion_text = concat_databases(emotion_final, text_emotion)
new_emotion_text["label"].replace({'happiness': 'happy'}, inplace=True)

label_encoding_trinary(new_emotion_text, [(new_emotion_text['label'] == "sadness") | (new_emotion_text['label'] == "worry") | (new_emotion_text['label'] == "anger") | (new_emotion_text['label'] == "hate"), (new_emotion_text['label'] == "neutral")| (new_emotion_text['label'] == "boredom") | (new_emotion_text['label'] == "surprise") | (new_emotion_text['label'] == "empty") | (new_emotion_text['label'] == "fear"), (new_emotion_text['label'] == "love") | (new_emotion_text['label'] == "fun") | (new_emotion_text['label'] == "happiness") | (new_emotion_text['label'] == "relief") | (new_emotion_text['label'] == "enthusiasm")], [0, 1, 2])

X_train15, X_test15, y_train15, y_test15 = variables_selection(new_emotion_text['text'], new_emotion_text['trinary_emotion'])
vectorizer15 = vectorize_tfidf_data(TfidfVectorizer, new_emotion_text['text'], None)
text_train15, text_test15 = transform_training_data(vectorizer15, X_train15, X_test15)
model15 = define_model(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), text_train15, y_train15)
y_pred15, dataframe15 = predict_model(model15, text_test15, X_test15)
accuracy12 = calculate_accuracy_score(y_test15, y_pred15)


#### Etape 3

X_svm, y_svm, X_test16, y_test16 = step3_svm_variables(emotion_final, text_emotion, new_list, new_list2)
vectorizer16 = vectorize_tfidf_data(TfidfVectorizer, X_svm, X_test16, 1)
text_train16, text_test16 = transform_training_data(vectorizer16, X_svm, X_test16)
model16 = define_model(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), text_train16, y_svm)
y_pred16, dataframe16 = predict_model(model16, text_test16, X_test16)
accuracy13 = calculate_accuracy_score(y_test16, y_pred16)

joblib.dump([dataframe13, accuracy10, dataframe14, accuracy11, dataframe15, accuracy12, dataframe16, accuracy13], filename_svm)
