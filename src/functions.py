# Import of all the libraries we need for our functions

import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc


def separator():
    """
        Function which separates the outputs in order to make the content more readable
    """
    print('------------------------------')

def plot_roc(y_test, y_proba):
    """
        Function which plots a ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Model (auc = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

##### Fonctions des modèles

### Fonctions de l'étape 1

# Import of the databases

def import_database():
    """
        Function which imports the needed databases
    """
    database1 = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_emotion_final.csv')
    database2 = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_text_emotion.csv')
    return database1, database2

# Label-encoding of the 'emotion' column

def label_encoding(database, conditions_list, value):
    """
        Function which encodes the emotion column to a binary classification by using the label-encoding
    """
    condition = conditions_list
    values = value
    database['binary_emotion'] = np.select(condition, values)
    return condition, values, database

def label_encoding_trinary(database, conditions_list, value):
    """
        Function which encodes the emotion column to a trinary classification by using the label-encoding
    """
    condition = conditions_list
    values = value
    database['trinary_emotion'] = np.select(condition, values)
    return condition, values, database

# Selection of the variables

def variables_selection(features, target_data):
    """
        Function which uses the train_test_split method in order to train and test our datas
    """
    X = features
    y = target_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

# Creation of the model

def vectorize_data(vector_method, data, data2,  indicator = 0):
    """
        Function which vectorizes the data with a specific method (CountVectorizer)
    """
    vector = vector_method(min_df=0, lowercase=False)
    vector.fit(data)
    if indicator != 0:
        vector.fit(data2)
    else :
        pass
    return vector

def vectorize_tfidf_data(vector_method, data, data2, indicator = 0):
    """
        Function which vectorizes the data with a specific method (TFIDF Vectorizer)
    """
    vector = vector_method(analyzer=lambda x: x)
    vector.fit(data)
    if indicator != 0:
        vector.fit(data2)
    else :
        pass
    return vector

def transform_training_data(vector, X_train, X_test):
    """
        Function which transforms our data in order to use our model on it
    """
    text_train = vector.transform(X_train)
    text_test = vector.transform(X_test)
    return text_train, text_test

def define_model(model, X, y):
    """
        Function which defines a model to train on our data by using features and a target value
    """
    classifier = model
    classifier.fit(X, y)
    return classifier

def calculate_score(model, X_test, y_test):
    """
        Function which calculates different metrics
    """
    score = model.score(X_test, y_test)
    return round(score, 4)

def predict_model(model, X_test, X_test2):
    """
        Function which predicts target values with test values and displays the predictions in a dataframe
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    dataframe = pd.DataFrame({'text':X_test2, 'predictions':y_pred, 'proba_predictions':y_proba})
    return y_pred, y_proba, dataframe

### Fonctions de l'étape 2

def concat_databases(database1, database2):
    """
        Function which concatenates two databases
    """
    emotion_final_extracted = database1[['text', 'label']]
    text_emotion_extracted = database2[['text', 'label']]
    new_emotion_text = pd.concat([emotion_final_extracted, text_emotion_extracted])
    return new_emotion_text

### Fonctions de l'étape 3

def step3_variables(database1, database2):
    """
        Function which selects the trained datas and the tested datas for our three first models (logistic regression, xgboost and random forest) 
    """
    X = database2['text']
    y = database2['binary_emotion']
    X_test = database1['text']
    y_test = database1['binary_emotion']
    return X, y, X_test, y_test

def step3_svm_variables(database1, database2, liste1, liste2):
    """
        Function which selects the trained datas and the tested datas for our SVM model 
    """
    X = liste2
    y = database2['trinary_emotion']
    X_test = liste1
    y_test = database1['trinary_emotion']
    return X, y, X_test, y_test

def step3_neuralnetwork_variables(database1, database2):
    """
        Function which selects the trained datas and the tested datas for our neural network 
    """
    X = database2['text']
    y = database2['trinary_emotion']
    X_test = database1['text']
    y_test = database1['trinary_emotion']
    return X, y, X_test, y_test

def calculate_accuracy_score(y_test, y_pred):
    """
        Function which calculates the part of predicted correct results 
    """
    acc = accuracy_score(y_test, y_pred)
    return round(acc, 5)

