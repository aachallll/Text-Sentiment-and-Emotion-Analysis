import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

print("Downloading necessary NLTK datasets...")
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

lem = WordNetLemmatizer()

def clean_text(text):
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"@\S+", "", txt)
    txt = re.sub(r"[^\w\s]", " ", txt)
    words = nltk.tokenize.word_tokenize(txt.lower())
    words = [lem.lemmatize(w, "v") for w in words]
    return ' '.join(words)

print("Loading dataset...")
df = pd.read_csv('text_emotion.csv')

# Only keep emotions we care about
valid_sentiments = ['worry', 'sadness', 'happiness', 'love', 'hate']
df = df[df['sentiment'].isin(valid_sentiments)]

print(f"Cleaning {len(df)} text samples...")
df['cleaned_content'] = df['content'].apply(clean_text)

print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_content'])
y = df['sentiment']

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
model.fit(X, y)

print("Saving models...")
models_dir = 'emotion/models'
os.makedirs(models_dir, exist_ok=True)

pickle.dump(vectorizer, open(os.path.join(models_dir, 'vectorizer.pickle'), 'wb'))
pickle.dump(model, open(os.path.join(models_dir, 'finalized_model.sav'), 'wb'))

print("Retraining process completed successfully!")
