import csv
import re
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import SearchHistory, Notification

from sentiment.views import twitter_service, dataset_service, model_service

def add_notification(request, message):
    user = request.user if request.user.is_authenticated else None
    Notification.objects.create(user=user, message=message)

def get_notifications_api(request):
    user = request.user if request.user.is_authenticated else None
    if not user:
        notifications = Notification.objects.filter(user=None).order_by('-timestamp')[:10]
    else:
        notifications = Notification.objects.filter(user=user).order_by('-timestamp')[:15]
        
    data = [{
        'id': n.id,
        'message': n.message,
        'is_read': n.is_read,
        'timestamp': n.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for n in notifications]
    
    unread_count = Notification.objects.filter(user=user, is_read=False).count() if user else Notification.objects.filter(user=None, is_read=False).count()
    return JsonResponse({'notifications': data, 'unread_count': unread_count})

def mark_notification_read_api(request, id):
    try:
        n = Notification.objects.get(id=id)
        n.is_read = True
        n.save()
        return JsonResponse({'status': 'success'})
    except Notification.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Not found'}, status=404)

def mark_all_read_api(request):
    user = request.user if request.user.is_authenticated else None
    Notification.objects.filter(user=user, is_read=False).update(is_read=True)
    return JsonResponse({'status': 'success'})

def delete_notification_api(request, id):
    try:
        n = Notification.objects.get(id=id)
        n.delete()
        return JsonResponse({'status': 'success'})
    except Notification.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Not found'}, status=404)

def choose_sentiment_or_emotion(request):
    history = []
    if request.user.is_authenticated:
        history = SearchHistory.objects.filter(user=request.user).order_by('-timestamp')[:10]
    return render(request, 'home/home.html', {'history': history})

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            add_notification(request, "Welcome to Tweet Analyzer! Your registration was successful.")
            return redirect('/')
    else:
        form = UserCreationForm()
    return render(request, 'home/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            add_notification(request, f"Welcome back, {user.username}! Login successful.")
            return redirect('/')
    else:
        form = AuthenticationForm()
    return render(request, 'home/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('/')

@login_required
def delete_history_view(request):
    SearchHistory.objects.filter(user=request.user).delete()
    add_notification(request, "Search history cleared successfully.")
    return redirect('/')

def analyze_topic_details(tweets):
    if not tweets:
        return {}
    
    total = len(tweets)
    pos_count = 0
    neu_count = 0
    neg_count = 0
    total_conf = 0
    
    emotions = {'worry': 0, 'happiness': 0, 'sadness': 0, 'love': 0, 'hate': 0}
    hashtags = {}
    keywords = {}
    mentions = {}
    
    total_likes = 0
    total_retweets = 0
    total_replies = 0
    
    for t in tweets:
        text = t['text']
        
        # Sentiment
        sentiment = t.get('sentiment')
        if not sentiment:
            sentiment, conf = model_service.analyze_sentiment(text)
            t['sentiment'] = sentiment
            t['confidence'] = conf
        else:
            conf = t.get('confidence', 80)
            
        if sentiment == 'Positive':
            pos_count += 1
        elif sentiment == 'Negative':
            neg_count += 1
        else:
            neu_count += 1
            
        total_conf += conf
        
        # Emotion mapping
        emotion = t.get('emotion')
        if not emotion:
            text_lower = text.lower()
            if any(w in text_lower for w in ['love', 'adore', 'heart']):
                emotion = 'love'
            elif any(w in text_lower for w in ['happy', 'glad', 'joy', 'awesome', 'great']):
                emotion = 'happiness'
            elif any(w in text_lower for w in ['sad', 'cry', 'gloomy', 'sorry']):
                emotion = 'sadness'
            elif any(w in text_lower for w in ['hate', 'angry', 'mad', 'scandalous', 'ugh']):
                emotion = 'hate'
            else:
                emotion = 'worry'
            t['emotion'] = emotion
            
        emotions[emotion.lower()] = emotions.get(emotion.lower(), 0) + 1
        
        # Engagement
        total_likes += t.get('likes', 0)
        total_retweets += t.get('retweets', 0)
        total_replies += t.get('replies', 0)
        
        # Parse hashtags, keywords, mentions
        for tag in re.findall(r'#\w+', text):
            tag_lower = tag.lower()
            hashtags[tag_lower] = hashtags.get(tag_lower, 0) + 1
            
        for user in re.findall(r'@\w+', text):
            user_lower = user.lower()
            mentions[user_lower] = mentions.get(user_lower, 0) + 1
            
        for word in re.findall(r'\b[a-zA-Z]{4,}\b', text):
            word_lower = word.lower()
            if word_lower not in ['http', 'https', 'with', 'that', 'this', 'your', 'about']:
                keywords[word_lower] = keywords.get(word_lower, 0) + 1
                
    top_tags = [item[0] for item in sorted(hashtags.items(), key=lambda x: x[1], reverse=True)[:5]]
    top_keys = [item[0] for item in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]]
    top_users = [item[0] for item in sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    dominant_emotion = max(emotions, key=emotions.get) if emotions else "worry"
    influential = sorted(tweets, key=lambda x: x.get('likes', 0) + x.get('retweets', 0), reverse=True)[:3]
    
    return {
        'total': total,
        'pos_pct': round((pos_count / total * 100), 1) if total > 0 else 0,
        'neu_pct': round((neu_count / total * 100), 1) if total > 0 else 0,
        'neg_pct': round((neg_count / total * 100), 1) if total > 0 else 0,
        'avg_conf': round(total_conf / total, 1) if total > 0 else 0,
        'dominant_emotion': dominant_emotion,
        'emotions': emotions,
        'top_hashtags': top_tags,
        'top_keywords': top_keys,
        'top_mentions': top_users,
        'likes': total_likes,
        'retweets': total_retweets,
        'replies': total_replies,
        'influential': influential,
        'tweets': tweets
    }

def generate_comparison_summary(topic_a, topic_b, metrics_a, metrics_b):
    pos_a = metrics_a.get('pos_pct', 0)
    pos_b = metrics_b.get('pos_pct', 0)
    
    keywords_a = ", ".join(metrics_a.get('top_keywords', [])[:3])
    keywords_b = ", ".join(metrics_b.get('top_keywords', [])[:3])
    
    if pos_a > pos_b:
        leader, leader_val = topic_a, pos_a
        follower, follower_val = topic_b, pos_b
    else:
        leader, leader_val = topic_b, pos_b
        follower, follower_val = topic_a, pos_a
        
    summary = (
        f"{leader} has a higher positive sentiment ({leader_val}%) than {follower} ({follower_val}%). "
        f"Most positive discussions about {topic_a} focus on keywords like {keywords_a}, "
        f"while conversations surrounding {topic_b} mainly involve {keywords_b}."
    )
    return summary

def compare_view(request):
    if request.method == 'POST':
        topic_a = request.POST.get('topic_a', '').strip()
        topic_b = request.POST.get('topic_b', '').strip()
        
        if not topic_a or not topic_b:
            add_notification(request, "Comparison failed: Topic fields were empty.")
            return render(request, 'home/compare.html', {'error_message': 'Both topic fields are required.'})
            
        if request.user.is_authenticated:
            SearchHistory.objects.create(user=request.user, query=f"{topic_a} vs {topic_b}", analysis_type='compare_dataset')

        source_indicator = "Kaggle Dataset (text_emotion.csv)"
        tweets_a = dataset_service.query_dataset(topic_a, limit=30)
        tweets_b = dataset_service.query_dataset(topic_b, limit=30)
        
        if not tweets_a or not tweets_b:
            add_notification(request, "Comparison failed: one or both topics yielded 0 records.")
            return render(request, 'home/compare.html', {
                'error_message': 'No matches found in the local database for one or both queries. Try using words like: happy, love, worry, sadness, hate.'
            })
            
        add_notification(request, f"Comparison dataset query completed for {topic_a} vs {topic_b}.")
            
        metrics_a = analyze_topic_details(tweets_a)
        metrics_b = analyze_topic_details(tweets_b)
        
        summary = generate_comparison_summary(topic_a, topic_b, metrics_a, metrics_b)
        add_notification(request, f"Comparison model training and evaluations completed for {topic_a} vs {topic_b}.")
        
        request.session['compare_topic_a'] = topic_a
        request.session['compare_topic_b'] = topic_b
        request.session['compare_metrics_a'] = metrics_a
        request.session['compare_metrics_b'] = metrics_b
        
        context = {
            'topic_a': topic_a,
            'topic_b': topic_b,
            'mode': mode,
            'source_indicator': source_indicator,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'summary': summary
        }
        return render(request, 'home/compare_result.html', context)
        
    return render(request, 'home/compare.html')

def export_compare_csv(request):
    topic_a = request.session.get('compare_topic_a', 'Topic A')
    topic_b = request.session.get('compare_topic_b', 'Topic B')
    metrics_a = request.session.get('compare_metrics_a', {})
    metrics_b = request.session.get('compare_metrics_b', {})
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="comparison_{topic_a}_vs_{topic_b}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([f"Comparison Report: {topic_a} vs {topic_b}"])
    writer.writerow([])
    writer.writerow(["Metric", f"{topic_a}", f"{topic_b}"])
    writer.writerow(["Total Tweets Analyzed", metrics_a.get('total', 0), metrics_b.get('total', 0)])
    writer.writerow(["Positive Sentiment %", metrics_a.get('pos_pct', 0), metrics_b.get('pos_pct', 0)])
    writer.writerow(["Neutral Sentiment %", metrics_a.get('neu_pct', 0), metrics_b.get('neu_pct', 0)])
    writer.writerow(["Negative Sentiment %", metrics_a.get('neg_pct', 0), metrics_b.get('neg_pct', 0)])
    writer.writerow(["Avg Confidence", metrics_a.get('avg_conf', 0), metrics_b.get('avg_conf', 0)])
    writer.writerow(["Dominant Emotion", metrics_a.get('dominant_emotion', ""), metrics_b.get('dominant_emotion', "")])
    writer.writerow(["Total Likes", metrics_a.get('likes', 0), metrics_b.get('likes', 0)])
    writer.writerow(["Total Retweets", metrics_a.get('retweets', 0), metrics_b.get('retweets', 0)])
    writer.writerow(["Total Replies", metrics_a.get('replies', 0), metrics_b.get('replies', 0)])
    
    add_notification(request, f"Comparison CSV exported successfully.")
    return response

def export_compare_excel(request):
    topic_a = request.session.get('compare_topic_a', 'Topic A')
    topic_b = request.session.get('compare_topic_b', 'Topic B')
    metrics_a = request.session.get('compare_metrics_a', {})
    metrics_b = request.session.get('compare_metrics_b', {})
    
    response = HttpResponse(content_type='application/vnd.ms-excel')
    response['Content-Disposition'] = f'attachment; filename="comparison_{topic_a}_vs_{topic_b}.xls"'
    
    writer = csv.writer(response, delimiter='\t')
    writer.writerow([f"Comparison Report: {topic_a} vs {topic_b}"])
    writer.writerow([])
    writer.writerow(["Metric", f"{topic_a}", f"{topic_b}"])
    writer.writerow(["Total Tweets Analyzed", metrics_a.get('total', 0), metrics_b.get('total', 0)])
    writer.writerow(["Positive Sentiment %", metrics_a.get('pos_pct', 0), metrics_b.get('pos_pct', 0)])
    writer.writerow(["Neutral Sentiment %", metrics_a.get('neu_pct', 0), metrics_b.get('neu_pct', 0)])
    writer.writerow(["Negative Sentiment %", metrics_a.get('neg_pct', 0), metrics_b.get('neg_pct', 0)])
    writer.writerow(["Avg Confidence", metrics_a.get('avg_conf', 0), metrics_b.get('avg_conf', 0)])
    writer.writerow(["Dominant Emotion", metrics_a.get('dominant_emotion', ""), metrics_b.get('dominant_emotion', "")])
    writer.writerow(["Total Likes", metrics_a.get('likes', 0), metrics_b.get('likes', 0)])
    writer.writerow(["Total Retweets", metrics_a.get('retweets', 0), metrics_b.get('retweets', 0)])
    writer.writerow(["Total Replies", metrics_a.get('replies', 0), metrics_b.get('replies', 0)])
    
    add_notification(request, f"Comparison Excel report exported successfully.")
    return response

def export_compare_pdf(request):
    topic_a = request.session.get('compare_topic_a', 'Topic A')
    topic_b = request.session.get('compare_topic_b', 'Topic B')
    metrics_a = request.session.get('compare_metrics_a', {})
    metrics_b = request.session.get('compare_metrics_b', {})
    
    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="comparison_{topic_a}_vs_{topic_b}_report.txt"'
    
    response.write(f"COMPARATIVE SENTIMENT & EMOTION REPORT: {topic_a} vs {topic_b}\n")
    response.write("="*70 + "\n\n")
    
    response.write(f"Metric".ljust(30) + f"{topic_a}".ljust(20) + f"{topic_b}".ljust(20) + "\n")
    response.write("-"*70 + "\n")
    response.write(f"Total Tweets Analyzed".ljust(30) + f"{metrics_a.get('total', 0)}".ljust(20) + f"{metrics_b.get('total', 0)}".ljust(20) + "\n")
    response.write(f"Positive Sentiment %".ljust(30) + f"{metrics_a.get('pos_pct', 0)}%".ljust(20) + f"{metrics_b.get('pos_pct', 0)}%".ljust(20) + "\n")
    response.write(f"Neutral Sentiment %".ljust(30) + f"{metrics_a.get('neu_pct', 0)}%".ljust(20) + f"{metrics_b.get('neu_pct', 0)}%".ljust(20) + "\n")
    response.write(f"Negative Sentiment %".ljust(30) + f"{metrics_a.get('neg_pct', 0)}%".ljust(20) + f"{metrics_b.get('neg_pct', 0)}%".ljust(20) + "\n")
    response.write(f"Avg Confidence".ljust(30) + f"{metrics_a.get('avg_conf', 0)}%".ljust(20) + f"{metrics_b.get('avg_conf', 0)}%".ljust(20) + "\n")
    response.write(f"Dominant Emotion".ljust(30) + f"{metrics_a.get('dominant_emotion', '')}".ljust(20) + f"{metrics_b.get('dominant_emotion', '')}".ljust(20) + "\n")
    response.write(f"Total Likes".ljust(30) + f"{metrics_a.get('likes', 0)}".ljust(20) + f"{metrics_b.get('likes', 0)}".ljust(20) + "\n")
    response.write(f"Total Retweets".ljust(30) + f"{metrics_a.get('retweets', 0)}".ljust(20) + f"{metrics_b.get('retweets', 0)}".ljust(20) + "\n")
    response.write(f"Total Replies".ljust(30) + f"{metrics_a.get('replies', 0)}".ljust(20) + f"{metrics_b.get('replies', 0)}".ljust(20) + "\n")
    
    add_notification(request, f"Comparison PDF report exported successfully.")
    return response
