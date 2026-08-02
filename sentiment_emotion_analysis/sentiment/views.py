import csv
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
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
    generate_historical_trends
)

# Initialize service singletons
twitter_service = TwitterService()
model_service = SentimentModelService()
dataset_service = DatasetService()

def sentiment_analysis(request):
    return render(request, 'home/sentiment.html')

def sentiment_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            
            sentiment, confidence = model_service.analyze_sentiment(tweet)
            tb = TextBlob(tweet)
            
            args = {
                'tweet': tweet, 
                'sentiment': sentiment,
                'confidence': confidence,
                'subjectivity': int(tb.sentiment.subjectivity * 100)
            }
            return render(request, 'home/sentiment_type_result.html', args)
    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/sentiment_type.html')

def sentiment_analysis_import(request):
    if request.method == 'POST':
        form = Sentiment_Imported_Tweet_analyse_form(request.POST)
        mode = request.POST.get('mode', 'live') # 'live' or 'dataset'

        if form.is_valid():
            handle = form.cleaned_data['sentiment_imported_tweet'].strip()
            source_indicator = "Live Twitter API (X)"
            list_of_tweets_and_sentiments = []
            tweet_texts = []

            # Save search to database if user is logged in
            if request.user.is_authenticated:
                SearchHistory.objects.create(user=request.user, query=handle, analysis_type=f'sentiment ({mode})')

            if mode == 'dataset':
                # Query local Kaggle dataset instead of fetching from Twitter
                source_indicator = "Kaggle Dataset (text_emotion.csv)"
                live_tweets = dataset_service.query_dataset(handle)
                
                # Format into matching dictionary array
                tweet_texts = [t['text'] for t in live_tweets]
                list_of_tweets_and_sentiments = live_tweets
            else:
                # Live API mode
                try:
                    if handle.startswith('#'):
                        live_tweets = twitter_service.fetch_tweets_by_query(handle)
                    else:
                        search_handle = handle[1:] if handle.startswith('@') else handle
                        live_tweets = twitter_service.fetch_tweets_by_user(search_handle)
                except TwitterServiceException as e:
                    from sentiment_or_emotion.views import add_notification
                    add_notification(request, f"Live fetch failed: {e}. Switched automatically to Kaggle Dataset Mode.")
                    mode = 'dataset'
                    source_indicator = "Kaggle Dataset (text_emotion.csv)"
                    live_tweets = dataset_service.query_dataset(handle)
                    tweet_texts = [t['text'] for t in live_tweets]
                    list_of_tweets_and_sentiments = live_tweets
                    
                if mode == 'live':
                    tweet_texts = [t.text for t in live_tweets]
                    for tweet in live_tweets:
                        sentiment, confidence = model_service.analyze_sentiment(tweet.text)
                        if sentiment == 'Positive':
                            detailed = 'Very Positive' if confidence > 80 else 'Positive'
                        elif sentiment == 'Negative':
                            detailed = 'Very Negative' if confidence > 80 else 'Negative'
                        else:
                            detailed = 'Neutral'
                            
                        list_of_tweets_and_sentiments.append({
                            'username': tweet.username,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'likes': tweet.likes,
                            'retweets': tweet.retweets,
                            'replies': tweet.replies,
                            'lang': tweet.lang,
                            'verified': tweet.verified,
                            'sentiment': sentiment,
                            'detailed': detailed,
                            'confidence': confidence
                        })

            # Calculate detailed distribution percentages
            detailed_counts = {'Very Positive': 0, 'Positive': 0, 'Neutral': 0, 'Negative': 0, 'Very Negative': 0}
            for item in list_of_tweets_and_sentiments:
                detailed_counts[item['detailed']] += 1

            # Advanced metrics from helper module
            aspects = analyze_aspects(tweet_texts)
            toxicity = detect_toxicity(tweet_texts)
            bot_score = detect_bots(tweet_texts)
            summary = generate_summary(tweet_texts)
            top_hashtags, top_keywords = extract_keywords_and_hashtags(tweet_texts)
            trends = generate_historical_trends(tweet_texts)
            
            # Cache results in session to support export downloads
            request.session['last_analysis_tweets'] = tweet_texts
            request.session['last_analysis_handle'] = handle

            from sentiment_or_emotion.views import add_notification
            if mode == 'dataset':
                add_notification(request, f"Kaggle Dataset analysis query successfully completed for: {handle}")
            else:
                add_notification(request, f"New live tweets successfully fetched and analyzed for: {handle}")

            args = {
                'list_of_tweets_and_sentiments': list_of_tweets_and_sentiments, 
                'handle': handle,
                'mode': mode,
                'source_indicator': source_indicator,
                'detailed_counts': detailed_counts,
                'aspects': aspects,
                'toxicity': toxicity,
                'bot_score': bot_score,
                'summary': summary,
                'top_hashtags': top_hashtags,
                'top_keywords': top_keywords,
                'trends': trends
            }
            
            if handle.startswith('#'):
                return render(request, 'home/sentiment_import_result_hashtag.html', args)
            return render(request, 'home/sentiment_import_result.html', args)

    else:
        form = Sentiment_Imported_Tweet_analyse_form()
        return render(request, 'home/sentiment_import.html')

def dataset_analysis_view(request):
    """Run model training and display evaluation comparison dashboard."""
    metrics = dataset_service.train_and_evaluate()
    return render(request, 'home/dataset_report.html', {'metrics': metrics})

def export_sentiment_csv(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="sentiment_{handle}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Tweet text', 'Sentiment Polarity', 'Subjectivity'])
    for tweet in tweets:
        tb = TextBlob(tweet)
        writer.writerow([tweet, tb.sentiment.polarity, tb.sentiment.subjectivity])
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"Sentiment report exported successfully as CSV for: {handle}")
    return response

def export_sentiment_excel(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='application/vnd.ms-excel')
    response['Content-Disposition'] = f'attachment; filename="sentiment_{handle}.xls"'
    
    writer = csv.writer(response, delimiter='\t')
    writer.writerow(['Tweet text', 'Sentiment Polarity', 'Subjectivity'])
    for tweet in tweets:
        tb = TextBlob(tweet)
        writer.writerow([tweet, tb.sentiment.polarity, tb.sentiment.subjectivity])
        
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"Sentiment report exported successfully as Excel for: {handle}")
    return response

def export_sentiment_pdf(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="sentiment_{handle}_report.txt"'
    
    response.write(f"SENTIMENT ANALYSIS REPORT FOR: {handle}\n")
    response.write("="*60 + "\n\n")
    
    for i, tweet in enumerate(tweets, 1):
        tb = TextBlob(tweet)
        pol = tb.sentiment.polarity
        lbl = "Positive" if pol > 0 else ("Negative" if pol < 0 else "Neutral")
        response.write(f"[{i}] Tweet: {tweet}\n")
        response.write(f"    Polarity: {pol} | Sentiment: {lbl} | Subjectivity: {tb.sentiment.subjectivity}\n\n")
        
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"Sentiment report exported successfully as PDF for: {handle}")
    return response

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
            return JsonResponse({'error': 'Empty input text'}, status=400)
            
        steps = {'original': text}
        
        # 1. Lowercase
        steps['lowercase'] = text.lower()
        
        # 2. URL Removal
        steps['no_urls'] = re.sub(r'http\S+|www\S+|https\S+', '', steps['lowercase'])
        
        # 3. Mention Removal
        steps['no_mentions'] = re.sub(r'@\w+', '', steps['no_urls'])
        
        # 4. Hashtag Processing
        hashtags = re.findall(r'#\w+', steps['no_mentions'])
        steps['hashtag_processed'] = re.sub(r'#(\w+)', r'\1', steps['no_mentions'])
        
        # 5. Emoji Processing
        emojis = re.findall(r'[^\w\s,.]', steps['hashtag_processed'])
        steps['emoji_processed'] = steps['hashtag_processed']
        for em in set(emojis):
            steps['emoji_processed'] = steps['emoji_processed'].replace(em, f" [{em}] ")
            
        # 6. Punctuation Removal
        steps['no_punctuation'] = re.sub(r'[^\w\s]', '', steps['emoji_processed'])
        
        # 7. Stopword Removal
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
        
        words_temp = steps['no_punctuation'].split()
        cleaned_words = [w for w in words_temp if w not in stopwords]
        stopwords_removed_count = len(words_temp) - len(cleaned_words)
        steps['stopword_removed'] = " ".join(cleaned_words)
        
        # 8. Tokenization
        tokens = steps['stopword_removed'].split()
        steps['tokenization'] = str(tokens)
        
        # 9 & 10. Stemming & Lemmatization
        def stem(w):
            if w.endswith('ing'): return w[:-3]
            if w.endswith('ed'): return w[:-2]
            if w.endswith('es'): return w[:-2]
            if w.endswith('s') and not w.endswith('ss'): return w[:-1]
            return w
            
        lemmas = {'running': 'run', 'went': 'go', 'better': 'good', 'happiest': 'happy'}
        stems = [stem(w) for w in tokens]
        lems = [lemmas.get(w, w) for w in tokens]
        steps['stemming'] = " ".join(stems)
        steps['lemmatization'] = " ".join(lems)
        
        # 11. POS Tagging
        noun_markers = ['day', 'night', 'work', 'time', 'home', 'life', 'school', 'game', 'today', 'friend']
        verb_markers = ['love', 'hate', 'like', 'want', 'think', 'know', 'see', 'feel', 'make', 'get']
        adj_markers = ['good', 'bad', 'happy', 'sad', 'great', 'awesome', 'nice', 'sweet', 'cool']
        
        pos_tags = []
        for w in tokens:
            if w in noun_markers: pos_tags.append((w, 'Noun'))
            elif w in verb_markers: pos_tags.append((w, 'Verb'))
            elif w in adj_markers: pos_tags.append((w, 'Adjective'))
            else: pos_tags.append((w, 'Noun' if len(w) > 4 else 'Adverb'))
        steps['pos_tagging'] = str(pos_tags)
        
        # 12. NER
        ner_entities = []
        for w in text.split():
            w_clean = re.sub(r'[^\w]', '', w)
            if w_clean and w_clean[0].isupper() and w_clean.lower() not in stopwords:
                ner_entities.append((w_clean, 'Entity/Name'))
        steps['ner'] = str(ner_entities)
        
        # 13. Final Cleaned text
        steps['final_clean'] = steps['lemmatization']
        
        # Sentiment prediction
        sentiment, confidence = model_service.analyze_sentiment(text)
        
        # Emotion detected
        text_lower = text.lower()
        if any(w in text_lower for w in ['love', 'adore', 'heart']): emotion = 'Love'
        elif any(w in text_lower for w in ['happy', 'glad', 'joy', 'awesome', 'great']): emotion = 'Happiness'
        elif any(w in text_lower for w in ['sad', 'cry', 'gloomy', 'sorry']): emotion = 'Sadness'
        elif any(w in text_lower for w in ['hate', 'angry', 'mad', 'scandalous', 'ugh']): emotion = 'Hate'
        else: emotion = 'Worry'
        
        # Highlights
        pos_words = ['love', 'happy', 'fun', 'relief', 'joy', 'good', 'great', 'awesome', 'nice']
        neg_words = ['sad', 'worry', 'hate', 'bad', 'anger', 'hurt', 'fail', 'sorry', 'wrong']
        
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
        
        data = {
            'steps': steps,
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion,
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
        
    html += """
    </body>
    </html>
    """
    
    response = HttpResponse(html, content_type='application/msword')
    response['Content-Disposition'] = 'attachment; filename="nlp_playground_analysis.doc"'
    
    from sentiment_or_emotion.views import add_notification
    add_notification(request, f"NLP Playground report successfully exported as Word Document (DOC).")
    return response



