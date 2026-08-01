import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentModelService:
    def __init__(self):
        self.use_transformer = False
        self.classifier = None
        
        # 1. Attempt to load Twitter-RoBERTa pipeline (State of the Art Transformer)
        try:
            from transformers import pipeline
            # Use cardiffnlp's RoBERTa sentiment model which is specifically trained on 58M tweets
            # device=-1 forces CPU execution, preventing CUDA memory out-of-memory errors
            self.classifier = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment", 
                device=-1
            )
            self.use_transformer = True
            print("Successfully initialized state-of-the-art Twitter-RoBERTa pipeline.")
        except Exception as e:
            print(f"Transformers not available or failed to load: {e}. Falling back to state-of-the-art VADER.")

        # 2. Setup VADER (Valence Aware Dictionary and sEntiment Reasoner) as a high-quality fallback
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            try:
                nltk.download('vader_lexicon', quiet=True)
            except Exception:
                pass
        self.vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """Analyze text and return (label, confidence_score_percentage)."""
        if self.use_transformer and self.classifier:
            try:
                result = self.classifier(text)[0]
                # RoBERTa labels: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
                label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
                label = label_map.get(result['label'], 'Neutral')
                confidence = int(result['score'] * 100)
                return label, confidence
            except Exception:
                # If transformer fails at runtime, fall through to VADER
                pass

        # VADER fallback execution
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        # Standard VADER classification thresholds
        if compound >= 0.05:
            label = 'Positive'
            confidence = int((compound * 50) + 50)
        elif compound <= -0.05:
            label = 'Negative'
            confidence = int((abs(compound) * 50) + 50)
        else:
            label = 'Neutral'
            # Estimate confidence of neutral label based on how low the positive/negative intensities are
            confidence = int((1.0 - abs(compound)) * 90)
            
        return label, min(max(confidence, 50), 98)
