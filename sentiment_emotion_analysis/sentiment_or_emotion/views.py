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


