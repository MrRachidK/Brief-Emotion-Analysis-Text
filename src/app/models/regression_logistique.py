# Import of all the libraries we need to work

import sys
from altair.vegalite.v4.api import value
sys.path.insert(0, "//home/apprenant/simplon_project/Brief-Emotion-Analysis-Text/")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing
from src.functions import plot_roc


### Fonctions de l'étape 1

# Import of the databases

def import_database():
    database1 = pd.read_csv('/home/apprenant/simplon_project/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_emotion_final.csv')
    database2 = pd.read_csv('/home/apprenant/simplon_project/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_text_emotion.csv')
    return database1, database2

# Label-encoding of the 'emotion' column

def label_encoding(database, conditions_list, value):
    condition = conditions_list
    values = value
    database['binary_emotion'] = np.select(condition, values)
    return condition, values, database

# Selection of the variables

def variables_selection(features, target_data):
    X = features
    y = target_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Creation of the model

def vectorize_data(vector_method, data, data2,  indicator = 0):
    vector = vector_method(min_df=0, lowercase=False)
    vector.fit(data)
    if indicator != 0:
        vector.fit(data2)
    else :
        pass
    return vector

def transform_training_data(model, X_train, X_test):
    text_train = model.transform(X_train)
    text_test = model.transform(X_test)
    return text_train, text_test

def define_model(model, X, y):
    classifier = model
    classifier.fit(X, y)
    return classifier

def calculate_score(model, X_test, y_test):
    score = model.score(X_test, y_test)
    return round(score, 4)


def predict_model(model, X_test, X_test2):
    y_pred = model.predict(X_test)
    dataframe = pd.DataFrame({'text':X_test2, 'predictions':y_pred})
    return y_pred, dataframe

### Fonctions de l'étape 2

def concat_databases():
    emotion_final_extracted = emotion_final[['text', 'label']]
    text_emotion_extracted = text_emotion[['text', 'label']]
    new_emotion_text = pd.concat([emotion_final_extracted, text_emotion_extracted])
    return new_emotion_text

### Fonctions de l'étape 3

def step3_variables():
    X = text_emotion['text']
    y = text_emotion['binary_emotion']
    X_test = emotion_final['text']
    y_test = emotion_final['binary_emotion']
    return X, y, X_test, y_test

def calculate_accuracy_score(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    return round(acc, 5)

#### Etape 1

emotion_final, text_emotion = import_database()

# Database 1

label_encoding(emotion_final, [(emotion_final['label'] == "sadness") | (emotion_final['label'] == "fear") | (emotion_final['label'] == "anger"), (emotion_final['label'] == "love") | (emotion_final['label'] == "surprise") | (emotion_final['label'] == "happy")], [0, 1])

X_train, X_test, y_train, y_test = variables_selection(emotion_final['text'], emotion_final['binary_emotion'])
vectorizer = vectorize_data(CountVectorizer, emotion_final['text'], None)
text_train, text_test = transform_training_data(vectorizer, X_train, X_test)
model = define_model(LogisticRegression(), text_train, y_train)

# Database 2

label_encoding(text_emotion, [(text_emotion['label'] == "empty") | (text_emotion['label'] == "sadness") | (text_emotion['label'] == "worry") | (text_emotion['label'] == "hate") | (text_emotion['label'] == "boredom") | (text_emotion['label'] == "anger"), (text_emotion['label'] == "enthusiasm") | (text_emotion['label'] == "neutral") | (text_emotion['label'] == "surprise") | (text_emotion['label'] == "love") | (text_emotion['label'] == "fun") | (text_emotion['label'] == "happiness") | (text_emotion['label'] == "relief")] ,[0, 1])

X_train2, X_test2, y_train2, y_test2 = variables_selection(text_emotion['text'], text_emotion['binary_emotion'])
vectorizer2 = vectorize_data(CountVectorizer, text_emotion['text'], None)
text_train2, text_test2 = transform_training_data(vectorizer2, X_train2, X_test2)
model2 = define_model(LogisticRegression(), text_train2, y_train2)

#### Etape 2

new_emotion_text = concat_databases()
new_emotion_text["label"].replace({'happiness': 'happy'}, inplace=True)

label_encoding(new_emotion_text, [(new_emotion_text['label'] == "empty") | (new_emotion_text['label'] == "sadness") | (new_emotion_text['label'] == "worry") | (new_emotion_text['label'] == "hate") | (new_emotion_text['label'] == "boredom") | (new_emotion_text['label'] == "anger") | new_emotion_text['label'] == "fear", (new_emotion_text['label'] == "enthusiasm") | (new_emotion_text['label'] == "neutral") | (new_emotion_text['label'] == "surprise") | (new_emotion_text['label'] == "love") | (new_emotion_text['label'] == "fun") | (new_emotion_text['label'] == "happy") | (new_emotion_text['label'] == "relief")] ,[0, 1])

X_train3, X_test3, y_train3, y_test3 = variables_selection(new_emotion_text['text'], new_emotion_text['binary_emotion'])
vectorizer3 = vectorize_data(CountVectorizer, new_emotion_text['text'], None)
text_train3, text_test3 = transform_training_data(vectorizer3, X_train3, X_test3)
model3 = define_model(LogisticRegression(), text_train3, y_train3)

#### Etape 3

X, y, X_test4, y_test4 = step3_variables()
vectorizer4 = vectorize_data(CountVectorizer, X, X_test4, 1)
text_train4, text_test4 = transform_training_data(vectorizer4, X, X_test4)
model4 = define_model(LogisticRegression(), text_train4, y)


