import csv
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from textblob import TextBlob

from .forms import Emotion_Typed_Tweet_analyse_form
from .emotion_analysis_code import emotion_analysis_code
from .forms import Emotion_Imported_Tweet_analyse_form
from sentiment_or_emotion.models import SearchHistory

from sentiment.twitter_service import (
    TwitterService, TwitterServiceException, 
    TwitterCredentialsError, TwitterRateLimitError, TwitterAPIError
)
from sentiment.sentiment_model_service import SentimentModelService
from sentiment.nlp_utils import (
    analyze_confidence, analyze_aspects, detect_toxicity, 
    detect_bots, generate_summary, extract_keywords_and_hashtags, 
    generate_historical_trends
)

# Initialize service singletons
twitter_service = TwitterService()
model_service = SentimentModelService()

def emotion_analysis(request):
    return redirect('/sentiment/type/')

def emotion_analysis_type(request):
    return redirect('/sentiment/type/')

def emotion_analysis_import(request):
    return redirect('/sentiment/import/')

    error_message = None
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        handle = request.POST.get('emotion_imported_tweet', '').strip()
        analyse = emotion_analysis_code()
        
        list_of_tweets_and_emotions = []
        tweet_texts = []
        detailed_counts = {'worry': 0, 'happiness': 0, 'sadness': 0, 'love': 0, 'hate': 0}
        
        if csv_file:
            if not csv_file.name.endswith('.csv'):
                error_message = "Invalid file format. Please upload a valid CSV file (.csv)."
                return render(request, 'home/emotion_import.html', {'error_message': error_message})
                
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if df.empty:
                    error_message = "The uploaded CSV file is empty."
                    return render(request, 'home/emotion_import.html', {'error_message': error_message})
                
                # Detect text column
                text_col = None
                for col in ['text', 'content', 'tweet', 'comment', 'message', 'text_emotion', 'tweet_text']:
                    if col in df.columns:
                        text_col = col
                        break
                if not text_col:
                    text_col = df.columns[0]
                    
                df = df.dropna(subset=[text_col])
                sample_rows = df.head(50)
                handle = csv_file.name
                
                for idx, row in sample_rows.iterrows():
                    raw_text = str(row[text_col])
                    emotion = analyse.predict_emotion(raw_text).lower()
                    
                    matched = False
                    for k in detailed_counts.keys():
                        if k in emotion:
                            detailed_counts[k] += 1
                            matched = True
                            break
                    if not matched:
                        if 'worry' in emotion or 'anxiety' in emotion:
                            detailed_counts['worry'] += 1
                        else:
                            detailed_counts['happiness'] += 1
                            
                    tweet_texts.append(raw_text)
                    list_of_tweets_and_emotions.append({
                        'username': row.get('username', row.get('author', f'User_{idx}')),
                        'text': raw_text,
                        'created_at': row.get('created_at', 'N/A'),
                        'likes': int(row.get('likes', 0)),
                        'retweets': int(row.get('retweets', 0)),
                        'replies': int(row.get('replies', 0)),
                        'lang': 'en',
                        'verified': False,
                        'emotion': emotion.capitalize()
                    })
                
                from sentiment_or_emotion.views import add_notification
                add_notification(request, f"Successfully uploaded and analyzed dataset for emotions: {csv_file.name}")
                
            except Exception as e:
                error_message = f"Error reading CSV: {e}"
                return render(request, 'home/emotion_import.html', {'error_message': error_message})
                
        elif handle:
            from sentiment.views import dataset_service
            source_indicator = "Kaggle Dataset (text_emotion.csv)"
            live_tweets = dataset_service.query_dataset(handle)
            if not live_tweets:
                error_message = f"No records matching '{handle}' were found in the dataset."
                return render(request, 'home/emotion_import.html', {'error_message': error_message})
                
            tweet_texts = [t['text'] for t in live_tweets]
            
            for t in live_tweets:
                raw_text = t['text']
                emotion = analyse.predict_emotion(raw_text).lower()
                
                matched = False
                for k in detailed_counts.keys():
                    if k in emotion:
                        detailed_counts[k] += 1
                        matched = True
                        break
                if not matched:
                    if 'worry' in emotion or 'anxiety' in emotion:
                        detailed_counts['worry'] += 1
                    else:
                        detailed_counts['happiness'] += 1
                        
                list_of_tweets_and_emotions.append({
                    'username': t['username'],
                    'text': raw_text,
                    'created_at': t['created_at'],
                    'likes': t['likes'],
                    'retweets': t['retweets'],
                    'replies': t['replies'],
                    'lang': 'en',
                    'verified': t['verified'],
                    'emotion': emotion.capitalize()
                })
                
            if request.user.is_authenticated:
                SearchHistory.objects.create(user=request.user, query=handle, analysis_type='emotion_dataset')
                
            from sentiment_or_emotion.views import add_notification
            add_notification(request, f"Dataset emotion analysis query completed for: {handle}")
            
        else:
            error_message = "Please input a search keyword or upload a CSV file."
            return render(request, 'home/emotion_import.html', {'error_message': error_message})

        # Advanced metrics
        aspects = analyze_aspects(tweet_texts)
        toxicity = detect_toxicity(tweet_texts)
        bot_score = detect_bots(tweet_texts)
        summary = generate_summary(tweet_texts)
        top_hashtags, top_keywords = extract_keywords_and_hashtags(tweet_texts)
        trends = generate_historical_trends(tweet_texts)
        
        request.session['last_analysis_tweets'] = tweet_texts
        request.session['last_analysis_handle'] = handle
        
        args = {
            'list_of_tweets_and_emotions': list_of_tweets_and_emotions, 
            'handle': handle,
            'detailed_counts': detailed_counts,
            'aspects': aspects,
            'toxicity': toxicity,
            'bot_score': bot_score,
            'summary': summary,
            'top_hashtags': top_hashtags,
            'top_keywords': top_keywords,
            'trends': trends
        }
        return render(request, 'home/emotion_import_result.html', args)

    else:
        form = Emotion_Imported_Tweet_analyse_form()
        return render(request, 'home/emotion_import.html')

def export_emotion_csv(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="emotion_{handle}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Tweet text', 'Sentiment Polarity', 'Subjectivity'])
    for tweet in tweets:
        tb = TextBlob(tweet)
        writer.writerow([tweet, tb.sentiment.polarity, tb.sentiment.subjectivity])
    return response

def export_emotion_excel(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='application/vnd.ms-excel')
    response['Content-Disposition'] = f'attachment; filename="emotion_{handle}.xls"'
    
    writer = csv.writer(response, delimiter='\t')
    writer.writerow(['Tweet text', 'Sentiment Polarity', 'Subjectivity'])
    for tweet in tweets:
        tb = TextBlob(tweet)
        writer.writerow([tweet, tb.sentiment.polarity, tb.sentiment.subjectivity])
    return response

def export_emotion_pdf(request):
    tweets = request.session.get('last_analysis_tweets', [])
    handle = request.session.get('last_analysis_handle', 'Analysis')
    
    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="emotion_{handle}_report.txt"'
    
    response.write(f"EMOTION ANALYSIS REPORT FOR: {handle}\n")
    response.write("="*60 + "\n\n")
    
    for i, tweet in enumerate(tweets, 1):
        tb = TextBlob(tweet)
        pol = tb.sentiment.polarity
        lbl = "Positive" if pol > 0 else ("Negative" if pol < 0 else "Neutral")
        response.write(f"[{i}] Tweet: {tweet}\n")
        response.write(f"    Polarity: {pol} | Sentiment: {lbl} | Subjectivity: {tb.sentiment.subjectivity}\n\n")
        
    return response