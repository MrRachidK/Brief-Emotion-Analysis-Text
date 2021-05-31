## Importation des librairies dont on a besoin

import os
import sys
from pathlib import Path
sys.path.insert(0, '/home/apprenant/Documents/Brief-Emotion-Analysis-Text/')
sys.path.append(Path(os.path.join(os.path.abspath(''), '../')).resolve().as_posix())
import pandas as pd
import spacy as sp
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.dataset import Dataset

## Premier jeu de données

dataset_path = Path('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_emotion_final.csv').resolve()
dataset = Dataset(dataset_path)
dataset.load()
dataset.preprocess_texts()

dataset.cleaned_data.head()

cleaned_dataset = dataset.cleaned_data

# Tokenisation

cleaned_dataset_list = cleaned_dataset['text'].to_list()

tokenizer_w = WhitespaceTokenizer()
token_list = []
for i in cleaned_dataset_list:
    tokenized_list = tokenizer_w.tokenize(i) 
    token_list.append(tokenized_list)

# Stemming

def stemmer_snowball(text_list):
    snowball = SnowballStemmer(language='english')
    stemmed_list = []
    for liste in text_list:
        stemmed_sentence = []
        for i in liste:
            i = snowball.stem(i)
            stemmed_sentence.append(i)
        stemmed_list.append(stemmed_sentence)
    return stemmed_list

new_list = stemmer_snowball(token_list)
print(new_list)

## Second jeu de données

dataset_path = Path('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_text_emotion.csv').resolve()
dataset = Dataset(dataset_path)
dataset.load()
dataset.preprocess_texts()

dataset.cleaned_data.head()

cleaned_dataset = dataset.cleaned_data

# Tokenisation

cleaned_dataset_list = cleaned_dataset['text'].to_list()

tokenizer_w = WhitespaceTokenizer()
token_list2 = []
for i in cleaned_dataset_list:
    tokenized_list = tokenizer_w.tokenize(i) 
    token_list2.append(tokenized_list)

# Stemming

new_list2 = stemmer_snowball(token_list2)
print(new_list2)

