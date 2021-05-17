# Import of the libraries and the databases we need to analyze
import sys
sys.path.insert(0, '/home/apprenant/Documents/Brief-Emotion-Analysis-Text/')
import pandas as pd 
from src.functions import separator

emotion_final = pd.read_csv("/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/01_raw/Emotion_final.csv")
text_emotion = pd.read_csv("/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/01_raw/text_emotion.csv")

# 1. Deleting of the useless columns

text_emotion = text_emotion.drop('author', axis=1)
print(text_emotion)
separator()

# 2. Handling of the duplicated rows

print(emotion_final.duplicated().value_counts())
emotion_final = emotion_final.drop_duplicates()
separator()
print(text_emotion.duplicated().value_counts())
separator()

# 3. Handling of the missing values

print(emotion_final.isna().value_counts())
separator()
print(text_emotion.isna().value_counts())
separator()

# 4. Checking of the value types

print(emotion_final.dtypes)
separator()
print(text_emotion.dtypes)
separator()

# 5. Conversion of object values

emotion_final['Text'] = emotion_final['Text'].astype(str)
emotion_final['Emotion'] = emotion_final['Emotion'].astype(str)

text_emotion['sentiment'] = text_emotion['sentiment'].astype(str)
text_emotion['content'] = text_emotion['content'].astype(str)

# 6. Creation of the cleaned data

emotion_final.to_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_emotion_final.csv', index=False)
text_emotion.to_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_text_emotion.csv', index=False)