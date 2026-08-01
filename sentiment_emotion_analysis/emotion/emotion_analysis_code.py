import pandas as pd
import numpy as np
import nltk
import re
import pickle
import itertools
from nltk.stem.wordnet import WordNetLemmatizer 
from django.conf import settings
import os

# tweet = 'Layin n bed with a headache  ughhhh...waitin on your call...'

class emotion_analysis_code():

    lem = WordNetLemmatizer()
    _vectorizer = None
    _model = None

    @classmethod
    def load_models(cls):
        if cls._vectorizer is None or cls._model is None:
            path_vec = os.path.join(settings.MODELS, 'vectorizer.pickle')
            path_model = os.path.join(settings.MODELS, 'finalized_model.sav')
            cls._vectorizer = pickle.load(open(path_vec, 'rb'))
            cls._model = pickle.load(open(path_model, 'rb'))

    def cleaning(self, text):
        txt = str(text)
        # Remove links
        txt = re.sub(r"http\S+", "", txt)
        # Remove usernames
        txt = re.sub(r"@\S+", "", txt)
        # Replace non-word characters with spaces
        txt = re.sub(r"[^\w\s]", " ", txt)
        # Tokenize
        words = nltk.tokenize.word_tokenize(txt.lower())
        # Lemmatize
        cleaned_words = [self.lem.lemmatize(w, "v") for w in words]
        
        if len(cleaned_words) == 0:
            return ["no", "text"]
        return cleaned_words

    def predict_emotion(self, tweet):
        self.load_models()
        tweet_in_pandas = pd.Series(' '.join(self.cleaning(tweet)))

        test = self._vectorizer.transform(tweet_in_pandas)
        predicted_sentiment = self._model.predict(test)
        final_sentiment = predicted_sentiment[0].lower().strip()
        
        # Capitalize for UI consistency
        if final_sentiment == 'worry':
            return 'Worry'
        elif final_sentiment == 'sadness':
            return 'Sadness'
        elif final_sentiment == 'happiness':
            return 'Happiness'
        elif final_sentiment == 'love':
            return 'Love'
        elif final_sentiment == 'hate':
            return 'Hate'
        else:
            return final_sentiment.capitalize()