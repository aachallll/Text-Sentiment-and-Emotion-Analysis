from django.urls import path
from . import views

app_name = 'sentiment_or_emotion'

from sentiment import views as sentiment_views

urlpatterns = [
    path('', views.choose_sentiment_or_emotion, name="choose_sentiment_or_emotion"),
    path('api/pos-tag/', sentiment_views.playground_pos_api, name="root_pos_tag_api"),
    path('api/ner/', sentiment_views.playground_ner_api, name="root_ner_api"),
    path('login/', views.login_view, name="login"),
    path('signup/', views.signup_view, name="signup"),
    path('logout/', views.logout_view, name="logout"),
    path('delete-history/', views.delete_history_view, name="delete_history"),
    path('api/notifications/', views.get_notifications_api, name="get_notifications_api"),
    path('api/notifications/mark-read/<int:id>/', views.mark_notification_read_api, name="mark_notification_read_api"),
    path('api/notifications/mark-all-read/', views.mark_all_read_api, name="mark_all_read_api"),
    path('api/notifications/delete/<int:id>/', views.delete_notification_api, name="delete_notification_api"),
]