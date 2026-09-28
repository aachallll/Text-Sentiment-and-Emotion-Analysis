# Text Sentiment and Emotion Analysis Using NLP and Machine Learning

A clean, reliable academic Natural Language Processing (NLP) and Machine Learning mini project designed to analyze, classify, and visualize text sentiments and multi-class emotions from sentences and dataset files.

---

## 📌 Project Overview

This project provides an end-to-end NLP system for evaluating textual sentiment polarities (**Positive**, **Neutral**, **Negative**) and fine-grained emotional tones (**Happiness**, **Love**, **Worry**, **Sadness**, **Hate**).

Built with **Django**, **Scikit-learn**, **NLTK**, and **TextBlob**, the system features robust input validation to prevent meaningless gibberish from being misclassified, an educational 10-stage text preprocessing pipeline, CSV dataset batch analysis with PDF export, and an interactive sentiment quiz.

---

## 🚀 Key Modules

### 1. ✍️ Manual Text Analysis
* Enter or paste any English sentence or social media text.
* Classifies **Sentiment** (`Positive`, `Neutral`, `Negative`) with confidence score.
* Identifies **Emotion** (`Happiness`, `Love`, `Worry`, `Sadness`, `Hate`) with confidence thresholding (avoids forcing "Worry" on neutral text).
* **Linguistic Validation Layer**: Rejects random gibberish strings (e.g., `vdyqwgfubecjbhjcvyew`, `asdfghjkl`, repetitive characters, or symbols) and displays a clear advisory alert: *"Please enter meaningful text or a valid sentence."*

### 2. 📂 CSV Dataset Analysis
* Upload any `.csv` file containing text comments or reviews.
* Automatically detects the appropriate text column (`text`, `content`, `tweet`, `comment`, `sentence`, `message`).
* Batch-evaluates sentiments and emotions.
* Displays interactive distribution donut and bar charts.
* Provides a one-click **Export PDF Report** for academic documentation.

### 3. ⚙️ NLP Preprocessing Pipeline (10 Educational Stages)
* Visualizes the step-by-step transformation of raw text into structured tokens:
  1. **Raw Text** (Input)
  2. **Lowercase Conversion**
  3. **URL & Mention Removal**
  4. **Punctuation / Special Characters Removal**
  5. **Emoji Processing**
  6. **Tokenization**
  7. **Stopword Removal**
  8. **Stemming**
  9. **Lemmatization**
  10. **Final Cleaned Text Output**
* Includes word count, character count, readability score, and token frequency bar chart.

### 4. 🎮 Interactive Sentiment Quiz
* Tests human sentiment discernment against trained model predictions.
* 10 randomized real-world questions with difficulty selection.
* Provides immediate feedback and contextual explanations for each classification.

### 5. 🌓 Clean Academic UI & Dark/Light Mode
* Clean, professional dashboard with key performance indicators.
* Dark / Light mode toggle preserved in local storage.
* Fully responsive layout built with Bootstrap.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Web Framework** | Python 3.10+, Django 6.0 |
| **NLP & Machine Learning** | Scikit-learn, NLTK, VADER Lexicon, TextBlob, Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap, Chart.js, FontAwesome |
| **Database** | SQLite3 |

---

## 📁 Project Structure

```text
Twitter-Sentiment-Emotion-Analysis-master/
├── README.md
└── sentiment_emotion_analysis/
    ├── manage.py
    ├── text_emotion.csv              # Kaggle Emotion Dataset
    ├── sentiment/                    # Core Sentiment & Preprocessing Module
    │   ├── views.py
    │   ├── urls.py
    │   ├── nlp_utils.py              # Validation layer & thresholded inference
    │   ├── sentiment_model_service.py
    │   ├── dataset_service.py
    │   └── templates/home/
    │       ├── sentiment_type.html
    │       ├── sentiment_import.html
    │       ├── sentiment_import_result.html
    │       ├── playground.html
    │       └── quiz.html
    ├── emotion/                      # Emotion routes (unified with sentiment)
    │   ├── views.py
    │   └── emotion_analysis_code.py
    ├── sentiment_or_emotion/         # Main app, base templates & auth views
    │   ├── views.py
    │   ├── urls.py
    │   └── templates/home/
    │       ├── base.html
    │       └── home.html
    └── sentiment_emotion_analysis/   # Project settings & URL routing
        ├── settings.py
        └── urls.py
```

---

## 💻 How to Run in VS Code

### Step 1: Open the Project
Open the project folder in VS Code:
```bash
File -> Open Folder -> Twitter-Sentiment-Emotion-Analysis-master
```

### Step 2: Open Terminal in VS Code
Press ``Ctrl + ` `` (or `Terminal -> New Terminal`).

### Step 3: Navigate to the Django Directory
```bash
cd sentiment_emotion_analysis
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or install core packages: `pip install django textblob nltk scikit-learn pandas`)*

Download required NLTK corpora (if prompted):
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('words')"
```

### Step 5: Start the Development Server
```bash
python manage.py runserver
```

### Step 6: Open the Application
Open your web browser and visit:
```
http://127.0.0.1:8000/
```
