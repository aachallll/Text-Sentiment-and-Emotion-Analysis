# Social Media Sentiment & Emotion Analysis System Using NLP and Machine Learning

A comprehensive, full-stack Natural Language Processing (NLP) and Machine Learning web application designed to analyze, classify, and visualize sentiments and fine-grained emotions from social media text and datasets.

---

## 📌 Project Overview

Understanding public opinion on social media platforms is crucial for brand monitoring, customer feedback, and public perception analysis. This system processes raw social media comments and structured datasets through an advanced NLP pipeline and machine learning models, offering real-time polarity classification, emotional breakdown, interactive visualizations, and automated report exports.

The application operates completely offline without external API rate limits or credential requirements.

---

## 🚀 Key Features

### 1. ✍️ Manual Text Analysis
* Enter any custom social-media comment, sentence, or tweet.
* Real-time prediction of **Sentiment** (`Positive`, `Neutral`, `Negative`) with a **Confidence Score (%)**.
* Detection of fine-grained **Emotions** (`Happiness`, `Love`, `Worry`, `Sadness`, `Hate`).

### 2. 📂 CSV Dataset Upload & Batch Analysis
* Upload any custom `.csv` dataset containing social media posts or comments.
* **Auto-Column Detection**: Automatically identifies columns like `text`, `content`, `tweet`, `comment`, etc.
* Processes records in batches and generates dynamic distribution charts and metrics.
* Supports searching the built-in offline dataset (`text_emotion.csv`).

### 3. ⚙️ 14-Stage NLP Preprocessing Playground
* An educational, visual workspace breaking down text preprocessing step-by-step:
  1. Original Text
  2. Lowercase Conversion
  3. URL Removal
  4. Mention (`@user`) Removal
  5. Hashtag (`#tag`) Processing
  6. Emoji Processing & Mapping
  7. Punctuation Removal
  8. Stopword Filtering
  9. Tokenization
  10. Stemming (Porter Stemmer)
  11. Lemmatization
  12. Parts of Speech (POS) Tagging
  13. Named Entity Recognition (NER)
  14. Final Cleaned Text Output
* Generates keyword frequency charts and readability metrics.

### 4. 🎮 Interactive Sentiment Quiz
* An interactive game with **Easy, Medium, and Hard** difficulties.
* Evaluates tweets sampled from the Kaggle dataset.
* Provides instant feedback comparing user answers against dataset labels and model predictions.
* Includes score calculation and a full answer review card.

### 5. 🤖 Context-Aware AI Chatbot Assistant
* Chatbot assistant answering questions grounded in the analyzed text and dataset context.
* Summarizes sentiments, explains dominant emotions, and retrieves representative positive/negative posts.

### 6. 📊 Model Evaluation & Benchmarks
* In-depth performance evaluation of trained ML models (Logistic Regression / Naive Bayes / VADER).
* Displays **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

### 7. 📄 Dual-Format Report Exports
* **Export PDF**: Generates a print-optimized document view that automatically launches the browser's PDF print dialogue.
* **Export Word Document**: Generates Microsoft Word-compatible (`.doc`) reports.

### 8. 🔔 History & Notification Center
* Tracks recent search queries with one-click clear options.
* Live dropdown notifications for analysis completions and export actions.
* Light/Dark mode UI theme switch with responsive design.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.10+, Django 5/6 |
| **NLP & ML** | Scikit-learn, NLTK, VADER Lexicon, TextBlob, Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap, jQuery, Chart.js, FontAwesome |
| **Database** | SQLite3 |

---

## 📁 Project Directory Structure

```text
Twitter-Sentiment-Emotion-Analysis-master/
├── README.md
├── sample text.txt
└── sentiment_emotion_analysis/
    ├── manage.py
    ├── db.sqlite3
    ├── text_emotion.csv              # Preloaded Kaggle emotion dataset
    ├── retrain_emotion_model.py      # Script for model training
    ├── emotion/                      # Emotion analysis application module
    │   ├── views.py
    │   ├── urls.py
    │   └── templates/home/
    ├── sentiment/                    # Sentiment & Playground application module
    │   ├── views.py
    │   ├── urls.py
    │   ├── dataset_service.py
    │   ├── sentiment_model_service.py
    │   ├── nlp_utils.py
    │   └── templates/home/
    │       ├── playground.html       # NLP Playground
    │       ├── quiz.html             # Interactive Quiz
    │       └── dataset_report.html   # Model Metrics
    └── sentiment_or_emotion/         # Core project settings and dashboard
        ├── views.py
        ├── models.py                 # SearchHistory & Notification models
        ├── urls.py
        └── templates/home/
            ├── base.html             # Common layout & navbar
            └── home.html             # Dashboard homepage
```

---

## 💻 Installation & Setup Guide

### 1. Prerequisites
Ensure you have Python installed (Python 3.9 to 3.12 recommended):
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/aachallll/Twitter-Sentiment-emotion-Analysis.git
cd Twitter-Sentiment-Emotion-Analysis-master
```

### 3. (Optional) Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install django scikit-learn nltk textblob pandas numpy
```

*(Optional) Download required NLTK tokenizers if prompted:*
```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
```

### 5. Apply Database Migrations
Navigate into the `sentiment_emotion_analysis` folder containing `manage.py`:
```bash
cd sentiment_emotion_analysis
python manage.py makemigrations
python manage.py migrate
```

### 6. Create Superuser (Admin Access)
```bash
python manage.py createsuperuser
```
*(Default development credentials if configured: `admin` / `admin123`)*

---

## ▶️ Running the Application

From the `sentiment_emotion_analysis` folder, start the local development server:

```bash
python manage.py runserver
```

Open your browser and navigate to:
👉 **`http://localhost:8000/`** (or `http://127.0.0.1:8000/`)

---

## 📖 How to Use the System

1. **Dashboard (`/`)**: View overview metrics, recent search history, and feature navigation cards.
2. **Manual Text Analysis (`/sentiment/type`)**: Type or paste any social media post and click *Analyze* to view polarity, confidence, and detected emotion.
3. **CSV Dataset Analysis (`/sentiment/import`)**:
   * Option 1: Search by topic/keyword in the preloaded Kaggle dataset.
   * Option 2: Upload your own `.csv` file to perform batch classification.
4. **NLP Playground (`/sentiment/playground/`)**: Enter text or load a sample tweet to visualize all 14 preprocessing stages and word frequencies.
5. **Sentiment Quiz (`/sentiment/quiz/`)**: Test your sentiment labeling skills across Easy, Medium, and Hard tweets.
6. **AI Assistant (`/sentiment/chatbot/`)**: Chat with the contextual AI agent regarding your active analysis.
7. **Model Evaluation (`/sentiment/dataset-analysis/`)**: Inspect Accuracy, F1-Scores, Confusion Matrices, and baseline performance comparisons.

---

## 🔒 Security & Offline Design
* **Zero API Dependency**: Does not require Twitter/X API tokens, keys, or internet connectivity to analyze text.
* **CSRF Protection**: All form submissions and AJAX requests adhere to Django's built-in CSRF token security.

---

## 📄 License
This project is open-source and developed for academic and educational purposes.
