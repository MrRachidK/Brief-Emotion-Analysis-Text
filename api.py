from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re

model = joblib.load("regression_logistique_model.joblib")
vectorizer = joblib.load("regression_logistique_vectorizer.joblib")

app = FastAPI()

emotions = ['émotion négative', 'émotion positive']

@app.get("/")
async def home():
    return {"Hello": "World"}

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def classify_message(model, message) :
    message = preprocessor(message)
    vectorized_message = vectorizer.transform([message])
    emotion = model.predict(vectorized_message)
    return {'emotion': emotions[int(emotion)]}

# @app.get('/predict_emotion/')
# async def detect_emotion_query(message: str):   
#     return classify_message(model, message)

@app.get('/predict_emotion/{message}')
async def detect_emotion_path(message: str):
    return classify_message(model, message)

