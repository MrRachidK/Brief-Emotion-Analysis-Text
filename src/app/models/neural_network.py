# Import of all the libraries we need to work

import sys
from altair.vegalite.v4.api import value
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.feature_extraction.text import TfidfTransformer

from src.functions import *
import joblib
filename_neuralnetwork = "neural_network.sav"

#### Etape 1

# Import of our databases
emotion_final, text_emotion = import_database()

# Database 1

# Encoding of the emotion column
label_encoding_trinary(emotion_final, [(emotion_final['label'] == "sadness") | (emotion_final['label'] == "anger"),  (emotion_final['label'] == "surprise") | (emotion_final['label'] == "fear") ,(emotion_final['label'] == "love") | (emotion_final['label'] == "happy")], [0, 1, 2])

# Division of our data by using train_test_split
X_train17, X_test17, y_train17, y_test17 = variables_selection(emotion_final['text'], emotion_final['trinary_emotion'])

X_train17 = X_train17.to_list()
y_train17 = y_train17.to_numpy()
X_test17 = X_test17.to_list()
y_test17 = y_test17.to_numpy()

# integer encode the documents
vs = 10000
enc_docs = [one_hot(d , vs) for d in X_train17]
enc_docs_test = [one_hot(d , vs) for d in X_test17]

# pad documents to a max length of 66 words
max_length = 66
p_docs = pad_sequences(enc_docs, maxlen = max_length, padding = 'post')
p_docs_test = pad_sequences(enc_docs_test, maxlen = max_length, padding = 'post')

# define the number of classes at the end of our neural network
y_train17 = keras.utils.to_categorical(y_train17, 3)
y_test17 = keras.utils.to_categorical(y_test17, 3)

# define an early stopping to avoid underfitting/overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# define the model
modelEmb = Sequential()
modelEmb.add(Embedding(vs, 500, input_length=max_length))
modelEmb.add(Flatten())
modelEmb.add(Dense(64, activation='relu'))
modelEmb.add(Dense(64, activation='relu'))
modelEmb.add(Dense(64, activation='relu'))
modelEmb.add(Dense(3, activation='softmax'))
# compile the model
modelEmb.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
# summarize the model
print(modelEmb.summary())
# fit the model
modelEmb.fit(p_docs, y_train17, epochs = 3, callbacks = [early_stopping], verbose=1, validation_data = (p_docs_test, y_test17))
# evaluate the model
score = modelEmb.evaluate(p_docs_test, y_test17, verbose=1)





# Database 2

# Encoding of the emotion column
label_encoding_trinary(text_emotion, [ (text_emotion['label'] == "sadness") | (text_emotion['label'] == "worry") | (text_emotion['label'] == "hate")  | (text_emotion['label'] == "anger"), (text_emotion['label'] == "boredom") | (text_emotion['label'] == "surprise") |(text_emotion['label'] == "empty") | (text_emotion['label'] == "neutral"), (text_emotion['label'] == "love") | (text_emotion['label'] == "fun") | (text_emotion['label'] == "happiness") | (text_emotion['label'] == "relief") | (text_emotion['label'] == "enthusiasm")], [0, 1, 2])

# Division of our data by using train_test_split
X_train18, X_test18, y_train18, y_test18 = variables_selection(text_emotion['text'], text_emotion['trinary_emotion'])

X_train18 = X_train18.to_list()
y_train18 = y_train18.to_numpy()
X_test18 = X_test18.to_list()
y_test18 = y_test18.to_numpy()

# integer encode the documents
vs = 10000
enc_docs2 = [one_hot(d , vs) for d in X_train18]
enc_docs_test2 = [one_hot(d , vs) for d in X_test18]

# pad documents to a max length of 66 words
max_length = 66
p_docs2 = pad_sequences(enc_docs2, maxlen = max_length, padding = 'post')
p_docs_test2 = pad_sequences(enc_docs_test2, maxlen = max_length, padding = 'post')

# define the number of classes at the end of our neural network
y_train18 = keras.utils.to_categorical(y_train18, 3)
y_test18 = keras.utils.to_categorical(y_test18, 3)

# define an early stopping to avoid underfitting/overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# define the model
modelEmb2 = Sequential()
modelEmb2.add(Embedding(vs, 500, input_length=max_length))
modelEmb2.add(Flatten())
modelEmb2.add(Dense(64, activation='relu'))
modelEmb2.add(Dense(64, activation='relu'))
modelEmb2.add(Dense(64, activation='relu'))
modelEmb2.add(Dense(3, activation='softmax'))
# compile the model
modelEmb2.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
# summarize the model
print(modelEmb2.summary())
# fit the model
modelEmb2.fit(p_docs2, y_train18, epochs = 3, callbacks = [early_stopping], verbose=1, validation_data = (p_docs_test2, y_test18))
# evaluate the model
score2 = modelEmb2.evaluate(p_docs_test2, y_test18, verbose=1)



#### Etape 2

# Concatenation of the two databases
new_emotion_text = concat_databases(emotion_final, text_emotion)
new_emotion_text["label"].replace({'happiness': 'happy'}, inplace=True)

# Encoding of the emotion column
label_encoding_trinary(new_emotion_text, [(new_emotion_text['label'] == "sadness") | (new_emotion_text['label'] == "worry") | (new_emotion_text['label'] == "anger") | (new_emotion_text['label'] == "hate"), (new_emotion_text['label'] == "neutral")| (new_emotion_text['label'] == "boredom") | (new_emotion_text['label'] == "surprise") | (new_emotion_text['label'] == "empty") | (new_emotion_text['label'] == "fear"), (new_emotion_text['label'] == "love") | (new_emotion_text['label'] == "fun") | (new_emotion_text['label'] == "happiness") | (new_emotion_text['label'] == "relief") | (new_emotion_text['label'] == "enthusiasm")], [0, 1, 2])

# Division of our data by using train_test_split
X_train19, X_test19, y_train19, y_test19 = variables_selection(new_emotion_text['text'], new_emotion_text['trinary_emotion'])

X_train19 = X_train19.to_list()
y_train19 = y_train19.to_numpy()
X_test19 = X_test19.to_list()
y_test19 = y_test19.to_numpy()

# integer encode the documents
vs = 10000
enc_docs3 = [one_hot(d , vs) for d in X_train19]
enc_docs_test3 = [one_hot(d , vs) for d in X_test19]

# pad documents to a max length of 66 words
max_length = 66
p_docs3 = pad_sequences(enc_docs3, maxlen = max_length, padding = 'post')
p_docs_test3 = pad_sequences(enc_docs_test3, maxlen = max_length, padding = 'post')

# define the number of classes at the end of our neural network
y_train19 = keras.utils.to_categorical(y_train19, 3)
y_test19 = keras.utils.to_categorical(y_test19, 3)

# define an early stopping to avoid underfitting/overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# define the model
modelEmb3 = Sequential()
modelEmb3.add(Embedding(vs, 500, input_length=max_length))
modelEmb3.add(Flatten())
modelEmb3.add(Dense(64, activation='relu'))
modelEmb3.add(Dense(64, activation='relu'))
modelEmb3.add(Dense(64, activation='relu'))
modelEmb3.add(Dense(3, activation='softmax'))
# compile the model
modelEmb3.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
# summarize the model
print(modelEmb3.summary())
# fit the model
modelEmb3.fit(p_docs3, y_train19, epochs = 3, callbacks = [early_stopping], verbose=1, validation_data = (p_docs_test3, y_test19))
# evaluate the model
score3 = modelEmb3.evaluate(p_docs_test3, y_test19, verbose=1)




#### Etape 3

# Selection of the trained datas and the tested datas
X_train20, y_train20, X_test20, y_test20 = step3_neuralnetwork_variables(emotion_final, text_emotion)

X_train20 = X_train20.to_list()
y_train20 = y_train20.to_numpy()
X_test20 = X_test20.to_list()
y_test20 = y_test20.to_numpy()

# integer encode the documents
vs = 10000
enc_docs4 = [one_hot(d , vs) for d in X_train20]
enc_docs_test4 = [one_hot(d , vs) for d in X_test20]

# pad documents to a max length of 66 words
max_length = 66
p_docs4 = pad_sequences(enc_docs4, maxlen = max_length, padding = 'post')
p_docs_test4 = pad_sequences(enc_docs_test4, maxlen = max_length, padding = 'post')

# define the number of classes at the end of our neural network
y_train20 = keras.utils.to_categorical(y_train20, 3)
y_test20 = keras.utils.to_categorical(y_test20, 3)

# define an early stopping to avoid underfitting/overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# define the model
modelEmb4 = Sequential()
modelEmb4.add(Embedding(vs, 500, input_length=max_length))
modelEmb4.add(Flatten())
modelEmb4.add(Dense(64, activation='relu'))
modelEmb4.add(Dense(64, activation='relu'))
modelEmb4.add(Dense(64, activation='relu'))
modelEmb4.add(Dense(3, activation='softmax'))
# compile the model
modelEmb4.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
# summarize the model
print(modelEmb4.summary())
# fit the model
modelEmb4.fit(p_docs4, y_train20, epochs = 3, callbacks = [early_stopping], verbose=1, validation_data = (p_docs_test4, y_test20))
# evaluate the model
score4 = modelEmb4.evaluate(p_docs_test4, y_test20, verbose=1)


# Saving of the useful objects for our Streamlit app
joblib.dump([score, score2, score3, score4], filename_neuralnetwork)
