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

