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
    return render(request, 'home/emotion.html')

def emotion_analysis_type(request):
    if request.method == 'POST':
        form = Emotion_Typed_Tweet_analyse_form(request.POST)
        analyse = emotion_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['emotion_typed_tweet']
            emotion = analyse.predict_emotion(tweet)
            confidence = analyze_confidence(tweet)
            
            args = {
                'tweet': tweet, 
                'emotion': emotion,
                'confidence': confidence
            }
            return render(request, 'home/emotion_type_result.html', args)
    else:
        form = Emotion_Typed_Tweet_analyse_form()
        return render(request, 'home/emotion_type.html')

def emotion_analysis_import(request):
    if request.method == 'POST':
        form = Emotion_Imported_Tweet_analyse_form(request.POST)
        analyse = emotion_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['emotion_imported_tweet'].strip()

            try:
                # Fetch live tweets using service layer (with error checks, no mock fallback)
                if handle.startswith('#'):
                    live_tweets = twitter_service.fetch_tweets_by_query(handle)
                else:
                    search_handle = handle[1:] if handle.startswith('@') else handle
                    live_tweets = twitter_service.fetch_tweets_by_user(search_handle)
            except TwitterServiceException as e:
                args = {
                    'form': form,
                    'error_message': str(e)
                }
                return render(request, 'home/emotion_import.html', args)

            # Save search to database if user is logged in
            if request.user.is_authenticated:
                SearchHistory.objects.create(user=request.user, query=handle, analysis_type='emotion')

            list_of_tweets_and_emotions = []
            detailed_counts = {'worry': 0, 'happiness': 0, 'sadness': 0, 'love': 0, 'hate': 0}
            tweet_texts = [t.text for t in live_tweets]
            
            for tweet in live_tweets:
                emotion = analyse.predict_emotion(tweet.text).lower()
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
                    'username': tweet.username,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.likes,
                    'retweets': tweet.retweets,
                    'replies': tweet.replies,
                    'lang': tweet.lang,
                    'verified': tweet.verified,
                    'emotion': emotion.capitalize()
                })

            # Advanced metrics
            aspects = analyze_aspects(tweet_texts)
            toxicity = detect_toxicity(tweet_texts)
            bot_score = detect_bots(tweet_texts)
            summary = generate_summary(tweet_texts)
            top_hashtags, top_keywords = extract_keywords_and_hashtags(tweet_texts)
            trends = generate_historical_trends(tweet_texts)
            
            # Cache to session for exporting reports
            request.session['last_analysis_tweets'] = tweet_texts
            request.session['last_analysis_handle'] = handle

            from sentiment_or_emotion.views import add_notification
            add_notification(request, f"New live tweets successfully fetched and emotion analyzed for: {handle}")

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
            
            if handle.startswith('#'):
                return render(request, 'home/emotion_import_result_hashtag.html', args)
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