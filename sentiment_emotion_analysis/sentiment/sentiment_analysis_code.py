import re
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer 
import itertools
import numpy as np
import nltk


class sentiment_analysis_code():

    lem = WordNetLemmatizer()

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

    def get_tweet_sentiment(self, tweet):
        # cleaning of tweet
        cleaned_tweet = ' '.join(self.cleaning(tweet))
        analysis = TextBlob(cleaned_tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'