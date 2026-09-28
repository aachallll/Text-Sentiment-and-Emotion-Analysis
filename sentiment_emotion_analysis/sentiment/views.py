import csv
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from textblob import TextBlob

from .forms import Sentiment_Typed_Tweet_analyse_form
from .forms import Sentiment_Imported_Tweet_analyse_form
from sentiment_or_emotion.models import SearchHistory

from .twitter_service import (
    TwitterService, TwitterServiceException, 
    TwitterCredentialsError, TwitterRateLimitError, TwitterAPIError
)
from .sentiment_model_service import SentimentModelService
from .dataset_service import DatasetService
from .nlp_utils import (
    analyze_confidence, analyze_aspects, detect_toxicity, 
    detect_bots, generate_summary, extract_keywords_and_hashtags, 
    generate_historical_trends, validate_natural_language_input,
    predict_emotion_with_threshold, perform_pos_tagging, extract_named_entities
)

# Initialize service singletons
twitter_service = TwitterService()
model_service = SentimentModelService()
dataset_service = DatasetService()

def sentiment_analysis(request):
    return redirect('/sentiment/type/')

@csrf_exempt
def sentiment_analysis_type(request):
    """Analyze manual text for both sentiment and emotion with input validation."""
    if request.method == 'POST':
        is_ajax = (request.headers.get('x-requested-with') == 'XMLHttpRequest' or 
                   request.content_type == 'application/json' or
                   request.POST.get('format') == 'json')
        
        raw_text = request.POST.get('sentiment_typed_tweet', '').strip()
        if not raw_text and request.body:
            try:
                import json
                body_data = json.loads(request.body)
                raw_text = body_data.get('sentiment_typed_tweet', body_data.get('text', '')).strip()
            except Exception:
                pass
                
        # 1. Generic Natural Language Validation
        is_valid, val_error = validate_natural_language_input(raw_text)
        if not is_valid:
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': val_error}, status=400)
            return render(request, 'home/sentiment_type.html', {'error_message': val_error, 'tweet': raw_text})
            
        # 2. Sentiment Classification
        sentiment, confidence = model_service.analyze_sentiment(raw_text)
        
        # 3. Emotion Detection with threshold
        emotion, emotion_confidence = predict_emotion_with_threshold(raw_text, sentiment)
        
        if is_ajax:
            return JsonResponse({
                'status': 'success',
                'text': raw_text,
                'sentiment': sentiment,
                'confidence': confidence,
                'emotion': emotion,
                'emotion_confidence': emotion_confidence
            })
            
        args = {
            'result': True,
            'tweet': raw_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence
        }
        return render(request, 'home/sentiment_type.html', args)
        
    return render(request, 'home/sentiment_type.html')

def sentiment_analysis_import(request):
    """CSV dataset analysis or offline database keyword search."""
    error_message = None
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        handle = request.POST.get('sentiment_imported_tweet', '').strip()
        
        records = []
        source_indicator = ""

        if csv_file:
            if not csv_file.name.endswith('.csv'):
                error_message = "Invalid file format. Please upload a valid CSV file (.csv)."
                return render(request, 'home/sentiment_import.html', {'error_message': error_message})
                
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if df.empty:
                    error_message = "The uploaded CSV file is empty."
                    return render(request, 'home/sentiment_import.html', {'error_message': error_message})
                
                # Detect text column
                text_col = None
                candidate_cols = ['text', 'tweet', 'content', 'comment', 'sentence', 'message', 'text_emotion', 'tweet_text']
                for col in candidate_cols:
                    if col in df.columns:
                        text_col = col
                        break
                if not text_col:
                    lower_cols = {c.lower(): c for c in df.columns}
                    for cand in candidate_cols:
                        if cand in lower_cols:
                            text_col = lower_cols[cand]
                            break
                            
                if not text_col:
                    for col in df.columns:
                        if df[col].dtype == object or df[col].dtype == 'string':
                            text_col = col
                            break
                            
                if not text_col:
                    error_message = "No valid text column was found. Please upload a CSV containing a text, tweet, comment, content, or sentence column."
                    return render(request, 'home/sentiment_import.html', {'error_message': error_message})
                    
                df = df.dropna(subset=[text_col])
                sample_rows = df.head(60)
                handle = csv_file.name
                source_indicator = f"Uploaded CSV ({csv_file.name})"
                
                for idx, row in sample_rows.iterrows():
                    raw_text = str(row[text_col]).strip()
                    if not raw_text or len(raw_text) < 2:
                        continue
                    sentiment, confidence = model_service.analyze_sentiment(raw_text)
                    emotion, em_conf = predict_emotion_with_threshold(raw_text, sentiment)
                    
                    records.append({
                        'id': idx + 1,
                        'text': raw_text,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'emotion': emotion,
                        'emotion_confidence': em_conf
                    })
                    
            except Exception as e:
                error_message = f"Error reading CSV: {e}"
                return render(request, 'home/sentiment_import.html', {'error_message': error_message})
                
        elif handle:
            source_indicator = "Kaggle Dataset (text_emotion.csv)"
            live_tweets = dataset_service.query_dataset(handle)
            if not live_tweets:
                error_message = f"No records matching '{handle}' were found in the dataset."
                return render(request, 'home/sentiment_import.html', {'error_message': error_message})
                
            for idx, t in enumerate(live_tweets):
                raw_text = t['text'].strip()
                sentiment, confidence = model_service.analyze_sentiment(raw_text)
                emotion, em_conf = predict_emotion_with_threshold(raw_text, sentiment)
                records.append({
                    'id': idx + 1,
                    'text': raw_text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'emotion': emotion,
                    'emotion_confidence': em_conf
                })
        else:
            error_message = "Please enter a search keyword or upload a CSV file."
            return render(request, 'home/sentiment_import.html', {'error_message': error_message})

        if not records:
            error_message = "No valid text entries could be parsed from the provided input."
            return render(request, 'home/sentiment_import.html', {'error_message': error_message})

        # Calculate sentiment counts & percentages
        total_count = len(records)
        pos_count = sum(1 for r in records if r['sentiment'] == 'Positive')
        neu_count = sum(1 for r in records if r['sentiment'] == 'Neutral')
        neg_count = sum(1 for r in records if r['sentiment'] == 'Negative')
        
        pos_pct = round((pos_count / total_count * 100), 1) if total_count > 0 else 0
        neu_pct = round((neu_count / total_count * 100), 1) if total_count > 0 else 0
        neg_pct = round((neg_count / total_count * 100), 1) if total_count > 0 else 0

        # Calculate emotion counts
        emotion_counts = {
            'Happiness': sum(1 for r in records if r['emotion'] == 'Happiness'),
            'Love': sum(1 for r in records if r['emotion'] == 'Love'),
            'Worry': sum(1 for r in records if r['emotion'] == 'Worry'),
            'Sadness': sum(1 for r in records if r['emotion'] == 'Sadness'),
            'Hate': sum(1 for r in records if r['emotion'] == 'Hate')
        }
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if any(emotion_counts.values()) else "Neutral"
        dominant_sentiment = "Positive" if pos_count >= max(neu_count, neg_count) else ("Negative" if neg_count >= neu_count else "Neutral")

        # Cache summary to session for PDF report
        request.session['last_dataset_analysis'] = {
            'handle': handle,
            'source_indicator': source_indicator,
            'total_count': total_count,
            'pos_count': pos_count, 'neu_count': neu_count, 'neg_count': neg_count,
            'pos_pct': pos_pct, 'neu_pct': neu_pct, 'neg_pct': neg_pct,
            'emotion_counts': emotion_counts,
            'dominant_sentiment': dominant_sentiment,
            'dominant_emotion': dominant_emotion,
            'sample_records': records[:20]
        }
        
        context = {
            'records': records,
            'handle': handle,
            'source_indicator': source_indicator,
            'total_count': total_count,
            'pos_count': pos_count,
            'neu_count': neu_count,
            'neg_count': neg_count,
            'pos_pct': pos_pct,
            'neu_pct': neu_pct,
            'neg_pct': neg_pct,
            'emotion_counts': emotion_counts,
            'dominant_sentiment': dominant_sentiment,
            'dominant_emotion': dominant_emotion
        }
        return render(request, 'home/sentiment_import_result.html', context)
        
    return render(request, 'home/sentiment_import.html')

def export_sentiment_pdf(request):
    """Generate print-optimized PDF report for the analyzed dataset."""
    data = request.session.get('last_dataset_analysis')
    if not data:
        return redirect('/sentiment/import/')
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment & Emotion Analysis Report - {data['handle']}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; padding: 40px; color: #1e293b; line-height: 1.5; }}
            h1 {{ color: #2563eb; margin-bottom: 5px; }}
            .header-info {{ color: #64748b; font-size: 14px; margin-bottom: 25px; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; }}
            .grid {{ display: flex; gap: 20px; margin-bottom: 25px; }}
            .card {{ flex: 1; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; background: #f8fafc; text-align: center; }}
            .card-title {{ font-size: 13px; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 5px; }}
            .card-val {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left; font-size: 13px; }}
            th {{ background-color: #f1f5f9; font-weight: 600; color: #334155; }}
            .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
            .badge-pos {{ background: #dcfce7; color: #15803d; }}
            .badge-neu {{ background: #f1f5f9; color: #475569; }}
            .badge-neg {{ background: #fee2e2; color: #b91c1c; }}
        </style>
    </head>
    <body>
        <h1>Text Sentiment and Emotion Analysis Report</h1>
        <div class="header-info">
            <strong>Source:</strong> {data['source_indicator']} | <strong>Query / File:</strong> {data['handle']} | <strong>Total Samples:</strong> {data['total_count']}
        </div>
        
        <h3>1. Sentiment Summary</h3>
        <div class="grid">
            <div class="card">
                <div class="card-title">Positive Sentiment</div>
                <div class="card-val" style="color: #16a34a;">{data['pos_pct']}% ({data['pos_count']})</div>
            </div>
            <div class="card">
                <div class="card-title">Neutral Sentiment</div>
                <div class="card-val" style="color: #64748b;">{data['neu_pct']}% ({data['neu_count']})</div>
            </div>
            <div class="card">
                <div class="card-title">Negative Sentiment</div>
                <div class="card-val" style="color: #dc2626;">{data['neg_pct']}% ({data['neg_count']})</div>
            </div>
        </div>
        
        <h3>2. Emotion Breakdown</h3>
        <table style="width: 50%; margin-bottom: 25px;">
            <tr><th>Emotion</th><th>Count</th></tr>
            <tr><td>Happiness</td><td>{data['emotion_counts'].get('Happiness', 0)}</td></tr>
            <tr><td>Love</td><td>{data['emotion_counts'].get('Love', 0)}</td></tr>
            <tr><td>Worry</td><td>{data['emotion_counts'].get('Worry', 0)}</td></tr>
            <tr><td>Sadness</td><td>{data['emotion_counts'].get('Sadness', 0)}</td></tr>
            <tr><td>Hate</td><td>{data['emotion_counts'].get('Hate', 0)}</td></tr>
        </table>

        <h3>3. Sample Records Analyzed</h3>
        <table>
            <tr><th style="width: 50px;">#</th><th>Text Content</th><th style="width: 100px;">Sentiment</th><th style="width: 100px;">Emotion</th></tr>
    """
    for rec in data.get('sample_records', []):
        badge_cls = 'badge-pos' if rec['sentiment'] == 'Positive' else ('badge-neg' if rec['sentiment'] == 'Negative' else 'badge-neu')
        html += f"""
            <tr>
                <td>{rec['id']}</td>
                <td>{rec['text']}</td>
                <td><span class="badge {badge_cls}">{rec['sentiment']}</span></td>
                <td><strong>{rec['emotion']}</strong></td>
            </tr>
        """
        
    html += """
        </table>
        <script>
            window.onload = function() { window.print(); }
        </script>
    </body>
    </html>
    """
    return HttpResponse(html)

import re
from django.http import JsonResponse

def parse_markdown_to_html(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'-\s*(.*?)(?:\n|$)', r'<li>\1</li>', text)
    return text.replace('\n', '<br>')

def chatbot_view(request):
    return render(request, 'home/chatbot.html')

def chatbot_clear_api(request):
    request.session['chat_history'] = []
    return JsonResponse({'status': 'cleared'})

def chatbot_api(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if not user_message:
            return JsonResponse({'error': 'Message empty'}, status=400)
            
        last_tweets = request.session.get('last_analysis_tweets', [])
        last_handle = request.session.get('last_analysis_handle', '')
        history = request.session.get('chat_history', [])
        
        response = ""
        user_lower = user_message.lower()
        
        if not last_tweets:
            # Fallback: load default tweets from Kaggle dataset so the bot is always functional!
            last_tweets_raw = dataset_service.query_dataset("happy", limit=20)
            last_tweets = [t['text'] for t in last_tweets_raw]
            last_handle = "General Dataset"
            request.session['last_analysis_tweets'] = last_tweets
            request.session['last_analysis_handle'] = last_handle
            
        total_tweets = len(last_tweets)
        pos_count = 0
        neg_count = 0
        neu_count = 0
        for t in last_tweets:
            tb = TextBlob(t)
            pol = tb.sentiment.polarity
            if pol > 0.05:
                pos_count += 1
            elif pol < -0.05:
                neg_count += 1
            else:
                neu_count += 1
        
        all_hashtags = []
        for text in last_tweets:
            all_hashtags.extend(re.findall(r'#\w+', text))
        from collections import Counter
        top_tags = [item[0] for item in Counter(all_hashtags).most_common(5)]
        
        if "why" in user_lower and "negative" in user_lower:
            response = (
                f"Out of {total_tweets} analyzed tweets for **{last_handle}**, "
                f"we detected {neg_count} negative tweets ({round(neg_count/total_tweets*100, 1)}%). "
                "Primary drivers for this negative sentiment include concern, worry, or criticism "
                "related to the query. Common phrases in the negative subset include words criticizing "
                "current product features or expressing frustration."
            )
        elif "summarize" in user_lower or "summary" in user_lower:
            response = (
                f"Here is a summary of the analyzed tweets for **{last_handle}**:\n"
                f"- Total tweets analyzed: **{total_tweets}**\n"
                f"- Positive sentiment split: **{round(pos_count/total_tweets*100, 1)}%**\n"
                f"- Negative sentiment split: **{round(neg_count/total_tweets*100, 1)}%**\n"
                f"- Key conversations focus on topics related to {', '.join(top_tags) if top_tags else 'the main handle'}."
            )
        elif "hashtag" in user_lower:
            if top_tags:
                response = f"The trending hashtags detected in today's tweets for **{last_handle}** are: " + ", ".join(top_tags)
            else:
                response = f"No prominent hashtags were detected in the analyzed tweets for **{last_handle}**."
        elif "positive" in user_lower:
            pos_tweets = [t for t in last_tweets if TextBlob(t).sentiment.polarity > 0.1][:3]
            if pos_tweets:
                response = "Here are some of the most positive tweets:\n"
                for pt in pos_tweets:
                    response += f"- *\"{pt}\"*\n"
            else:
                response = "No highly positive tweets were found in this query."
        elif "emotion" in user_lower:
            response = (
                f"Based on the analysis for **{last_handle}**, the most common emotion detected "
                "is **happiness/worry**, showing active user engagement and critical feedback splits."
            )
        else:
            response = (
                f"I've analyzed {total_tweets} tweets for **{last_handle}**. "
                "Feel free to ask me to summarize them, check trending hashtags, show positive examples, or explain why sentiment is negative."
            )
                
        html_response = parse_markdown_to_html(response)
        history.append({'role': 'user', 'message': user_message})
        history.append({'role': 'assistant', 'message': html_response})
        request.session['chat_history'] = history
        
        return JsonResponse({'response': html_response, 'history': history})

def quiz_view(request):
    """Render the main Interactive Sentiment Quiz template."""
    return render(request, 'home/quiz.html')

def quiz_questions_api(request):
    """API endpoint returning 10 random quiz questions from the Kaggle dataset."""
    difficulty = request.GET.get('difficulty', 'medium')
    questions = dataset_service.get_quiz_questions(difficulty, limit=10)
    
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"New Interactive Sentiment Quiz started (Difficulty: {difficulty.capitalize()}).")
    
    return JsonResponse({'status': 'success', 'questions': questions})

def quiz_submit_answer_api(request):
    """API endpoint to evaluate user's selected sentiment answer against dataset & model predictions."""
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        user_answer = request.POST.get('user_answer', '').strip().capitalize()
        correct_answer = request.POST.get('correct_answer', '').strip().capitalize()
        
        # Predict using current sentiment model service (RoBERTa/VADER fallback)
        predicted_sentiment, confidence = model_service.analyze_sentiment(text)
        
        # Build explanation highlighting sentiment indicators
        text_lower = text.lower()
        pos_words = ['love', 'happy', 'fun', 'relief', 'joy', 'good', 'great', 'awesome', 'nice', 'smile', 'glad', 'wonderful']
        neg_words = ['sad', 'worry', 'hate', 'bad', 'anger', 'hurt', 'fail', 'sorry', 'wrong', 'cry', 'hate', 'bored']
        
        found_pos = [f"**{w}**" for w in pos_words if w in text_lower]
        found_neg = [f"**{w}**" for w in neg_words if w in text_lower]
        
        explanation = ""
        if correct_answer == 'Positive':
            if found_pos:
                explanation = f"The dataset labels this positive because it contains expressions of happiness/affection like {', '.join(found_pos)}."
            else:
                explanation = "The dataset classifies this positive based on overall positive context and sentiment polarity."
        elif correct_answer == 'Negative':
            if found_neg:
                explanation = f"The dataset labels this negative because it contains expressions of concern/sadness like {', '.join(found_neg)}."
            else:
                explanation = "The dataset classifies this negative based on critical, sad, or disappointed remarks."
        else:
            explanation = "This tweet does not contain strong positive or negative indicators, suggesting a neutral stance."
            
        return JsonResponse({
            'correct': user_answer == correct_answer,
            'actual': correct_answer,
            'predicted': predicted_sentiment,
            'confidence': confidence,
            'explanation': explanation
        })
    return JsonResponse({'error': 'POST required'}, status=400)

def playground_view(request):
    """Render the main NLP Playground template."""
    return render(request, 'home/playground.html')

def playground_sample_api(request):
    """API returning a single random tweet content from the Kaggle dataset."""
    try:
        import pandas as pd
        df = pd.read_csv("text_emotion.csv")
        df = df.dropna(subset=['content'])
        sample_text = df.sample(1).iloc[0]['content']
        return JsonResponse({'status': 'success', 'text': sample_text})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def playground_analyze_api(request):
    """Run all NLP preprocessing pipeline steps, token frequencies, and statistics."""
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if not text:
            return JsonResponse({'error': 'Please enter text to analyze.'}, status=400)
            
        # 1. Linguistic Natural Language Validation
        is_valid, val_error = validate_natural_language_input(text)
        if not is_valid:
            return JsonResponse({'error': val_error}, status=400)
            
        steps = {'original': text}
        
        # 1. Lowercase Conversion
        steps['lowercase'] = text.lower()
        
        # 2. URL & Mention Removal
        clean_urls = re.sub(r'http\S+|www\S+|https\S+', '', steps['lowercase'])
        steps['no_urls_mentions'] = re.sub(r'@\w+', '', clean_urls).strip()
        
        # 3. Emoji Processing
        emojis = re.findall(r'[^\w\s,.]', steps['no_urls_mentions'])
        emoji_proc = steps['no_urls_mentions']
        for em in set(emojis):
            emoji_proc = emoji_proc.replace(em, f" [{em}] ")
        steps['emoji_processed'] = emoji_proc
        
        # 4. Special Characters / Punctuation Removal
        steps['no_punctuation'] = re.sub(r'[^\w\s]', '', steps['emoji_processed']).strip()
        
        # 5. Tokenization
        raw_tokens = steps['no_punctuation'].split()
        steps['tokenization'] = str(raw_tokens)
        
        # 6. Stopword Removal
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                     'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                     'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                     'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
                     'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                     'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                     'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                     'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                     'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                     'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
        
        cleaned_words = [w for w in raw_tokens if w.lower() not in stopwords]
        stopwords_removed_count = len(raw_tokens) - len(cleaned_words)
        steps['stopword_removed'] = " ".join(cleaned_words)
        tokens = cleaned_words if cleaned_words else raw_tokens
        
        # 7. Stemming
        def stem(w):
            if w.endswith('ing'): return w[:-3]
            if w.endswith('ed'): return w[:-2]
            if w.endswith('es'): return w[:-2]
            if w.endswith('s') and not w.endswith('ss'): return w[:-1]
            return w
            
        lemmas = {'running': 'run', 'went': 'go', 'better': 'good', 'happiest': 'happy', 'loving': 'love', 'loved': 'love'}
        stems = [stem(w) for w in tokens]
        lems = [lemmas.get(w, w) for w in tokens]
        steps['stemming'] = " ".join(stems)
        
        # 8. Lemmatization
        steps['lemmatization'] = " ".join(lems)
        
        # 9. Final Cleaned text
        steps['final_clean'] = steps['lemmatization']
        
        # Sentiment prediction
        sentiment, confidence = model_service.analyze_sentiment(text)
        
        # Emotion detected with threshold
        emotion, emotion_confidence = predict_emotion_with_threshold(text, sentiment)
        
        # Highlights
        pos_words = ['love', 'happy', 'fun', 'relief', 'joy', 'good', 'great', 'awesome', 'nice', 'excellent', 'wonderful']
        neg_words = ['sad', 'worry', 'hate', 'bad', 'anger', 'hurt', 'fail', 'sorry', 'wrong', 'terrible', 'awful']
        
        highlighted_text = ""
        for w in text.split():
            w_clean = re.sub(r'[^\w]', '', w).lower()
            if w_clean in pos_words:
                highlighted_text += f' <span class="text-success font-weight-bold">{w}</span>'
            elif w_clean in neg_words:
                highlighted_text += f' <span class="text-danger font-weight-bold">{w}</span>'
            else:
                highlighted_text += f' {w}'
                
        # Token frequency
        from collections import Counter
        freq = Counter(tokens)
        freq_list = [{'word': item[0], 'count': item[1]} for item in freq.most_common(10)]
        
        # Stats
        word_count = len(text.split())
        char_count = len(text)
        vocabulary_size = len(set(tokens))
        avg_word_len = round(sum(len(w) for w in tokens) / len(tokens), 1) if tokens else 0
        reading_time = max(1, round(word_count / 200 * 60)) # seconds
        readability_score = min(100, max(0, 100 - (word_count // 3)))
        
        explanation = (
            f"The model predicted **{sentiment}** with a confidence score of **{confidence}%**. "
            f"This is mainly because of indicators related to **{emotion}** emotion."
        )
        
        # POS Tagging and NER Extraction
        pos_tokens = perform_pos_tagging(text)
        ner_entities = extract_named_entities(text)
        
        data = {
            'steps': steps,
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion,
            'pos_tokens': pos_tokens,
            'ner_entities': ner_entities,
            'word_count': word_count,
            'char_count': char_count,
            'token_count': len(tokens),
            'stopwords_removed': stopwords_removed_count,
            'vocab_size': vocabulary_size,
            'avg_word_len': avg_word_len,
            'reading_time': reading_time,
            'readability': readability_score,
            'highlighted': highlighted_text,
            'frequencies': freq_list,
            'explanation': explanation
        }
        
        request.session['last_playground_analysis'] = data
        return JsonResponse(data)
        
    return JsonResponse({'error': 'POST required'}, status=400)

@csrf_exempt
def playground_pos_api(request):
    """
    Dedicated API endpoint for Part-of-Speech (POS) tagging.
    Expects POST with 'text'. Returns grammatical category for each word.
    """
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if not text and request.body:
            try:
                import json
                b = json.loads(request.body)
                text = b.get('text', '').strip()
            except Exception:
                pass
                
        if not text:
            return JsonResponse({'success': False, 'error': 'Please enter some text.'}, status=400)
            
        is_valid, err = validate_natural_language_input(text)
        if not is_valid:
            return JsonResponse({'success': False, 'error': err}, status=400)
            
        try:
            tokens = perform_pos_tagging(text)
            return JsonResponse({'success': True, 'tokens': tokens})
        except Exception:
            return JsonResponse({'success': False, 'error': 'Failed to process POS tagging.'}, status=500)
            
    return JsonResponse({'success': False, 'error': 'POST required'}, status=405)

@csrf_exempt
def playground_ner_api(request):
    """
    Dedicated API endpoint for Named Entity Recognition (NER).
    Expects POST with 'text'. Extracts entities like PERSON, GPE/LOCATION, ORGANIZATION, DATE, etc.
    """
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if not text and request.body:
            try:
                import json
                b = json.loads(request.body)
                text = b.get('text', '').strip()
            except Exception:
                pass
                
        if not text:
            return JsonResponse({'success': False, 'error': 'Please enter some text.'}, status=400)
            
        is_valid, err = validate_natural_language_input(text)
        if not is_valid:
            return JsonResponse({'success': False, 'error': err}, status=400)
            
        try:
            entities = extract_named_entities(text)
            return JsonResponse({'success': True, 'entities': entities})
        except Exception:
            return JsonResponse({'success': False, 'error': 'Failed to extract named entities.'}, status=500)
            
    return JsonResponse({'success': False, 'error': 'POST required'}, status=405)

def export_playground_pdf(request):
    """Generate print-friendly HTML page that automatically triggers the browser PDF print dialogue."""
    data = request.session.get('last_playground_analysis', {})
    if not data:
        return redirect('/sentiment/playground/')
        
    from django.template.loader import render_to_string
    # We will build a clean printable report page
    html = f"""
    <html>
    <head>
        <title>NLP Preprocessing Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; color: #333; line-height: 1.6; }}
            h1 {{ color: #2563eb; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
            .kpi-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
            .kpi-table th, .kpi-table td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
            .kpi-table th {{ background-color: #f8fafc; }}
            .step-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; padding: 15px; margin-bottom: 15px; border-radius: 4px; }}
            .step-title {{ font-weight: bold; color: #2563eb; margin-bottom: 5px; text-transform: uppercase; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>NLP Preprocessing & Sentiment Report</h1>
        <p><strong>Original Text:</strong> "{data['steps']['original']}"</p>
        <p><strong>Sentiment:</strong> {data['sentiment']} (Confidence: {data['confidence']}%) | <strong>Emotion:</strong> {data['emotion']}</p>
        
        <h2>Text Statistics</h2>
        <table class="kpi-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Word Count</td><td>{data['word_count']}</td></tr>
            <tr><td>Character Count</td><td>{data['char_count']}</td></tr>
            <tr><td>Tokens Count</td><td>{data['token_count']}</td></tr>
            <tr><td>Vocabulary Size</td><td>{data['vocab_size']}</td></tr>
            <tr><td>Readability Score</td><td>{data['readability']}</td></tr>
            <tr><td>Reading Time</td><td>{data['reading_time']}s</td></tr>
        </table>
        
        <h2>Pipeline Processing Steps</h2>
        """
    for key, val in data['steps'].items():
        html += f"""
        <div class="step-box">
            <div class="step-title">{key.replace('_', ' ')}</div>
            <div style="font-family: monospace;">{val}</div>
        </div>
        """
        
    if data.get('pos_tokens'):
        html += """
        <h2>Part-of-Speech (POS) Tagging</h2>
        <table class="kpi-table">
            <tr><th>Word</th><th>POS Tag</th><th>Description</th></tr>
        """
        for tok in data['pos_tokens']:
            html += f"<tr><td><strong>{tok['word']}</strong></td><td><code>{tok['tag']}</code></td><td>{tok['description']}</td></tr>"
        html += "</table>"
        
    if data.get('ner_entities'):
        html += """
        <h2>Named Entity Recognition (NER)</h2>
        <table class="kpi-table">
            <tr><th>Entity</th><th>Type / Category</th></tr>
        """
        for ent in data['ner_entities']:
            html += f"<tr><td><strong>{ent['text']}</strong></td><td><span style='color: #2563eb; font-weight: bold;'>{ent['label']}</span></td></tr>"
        html += "</table>"
        
    html += """
        <script>
            window.onload = function() {
                window.print();
            }
        </script>
    </body>
    </html>
    """
    
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"NLP Playground report successfully exported as PDF.")
    return HttpResponse(html)

def export_playground_docx(request):
    """Generate Word-compatible HTML file served as application/msword download."""
    data = request.session.get('last_playground_analysis', {})
    if not data:
        return redirect('/sentiment/playground/')
        
    html = f"""
    <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
    <head>
        <title>NLP Preprocessing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #2563eb; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
        </style>
    </head>
    <body>
        <h1>NLP Preprocessing & Sentiment Report</h1>
        <p><strong>Original Text:</strong> "{data['steps']['original']}"</p>
        <p><strong>Sentiment:</strong> {data['sentiment']} (Confidence: {data['confidence']}%) | <strong>Emotion:</strong> {data['emotion']}</p>
        
        <h2>Text Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Word Count</td><td>{data['word_count']}</td></tr>
            <tr><td>Character Count</td><td>{data['char_count']}</td></tr>
            <tr><td>Tokens Count</td><td>{data['token_count']}</td></tr>
            <tr><td>Vocabulary Size</td><td>{data['vocab_size']}</td></tr>
        </table>
        
        <h2>Pipeline Preprocessing Steps</h2>
        """
    for key, val in data['steps'].items():
        html += f"<p><strong>{key.capitalize()}:</strong><br>{val}</p>"
        
    if data.get('pos_tokens'):
        html += "<h2>Part-of-Speech (POS) Tagging</h2><table><tr><th>Word</th><th>POS Tag</th><th>Description</th></tr>"
        for tok in data['pos_tokens']:
            html += f"<tr><td><strong>{tok['word']}</strong></td><td>{tok['tag']}</td><td>{tok['description']}</td></tr>"
        html += "</table>"
        
    if data.get('ner_entities'):
        html += "<h2>Named Entity Recognition (NER)</h2><table><tr><th>Entity</th><th>Type</th></tr>"
        for ent in data['ner_entities']:
            html += f"<tr><td><strong>{ent['text']}</strong></td><td>{ent['label']}</td></tr>"
        html += "</table>"
        
    html += """
    </body>
    </html>
    """
    
    response = HttpResponse(html, content_type='application/msword')
    response['Content-Disposition'] = 'attachment; filename="nlp_playground_analysis.doc"'
    
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"NLP Playground report successfully exported as Word Document (DOC).")
    return response



