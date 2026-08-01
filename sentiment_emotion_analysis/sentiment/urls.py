from django.urls import path
from . import views

app_name = 'sentiment'

urlpatterns = [
    path('', views.sentiment_analysis, name="sentiment_anaylsis"),
    path('type/', views.sentiment_analysis_type, name="sentiment_analysis_type"),
    path('import/', views.sentiment_analysis_import, name="sentiment_analysis_import"),
    path('export-csv/', views.export_sentiment_csv, name="export_csv"),
    path('export-excel/', views.export_sentiment_excel, name="export_excel"),
    path('export-pdf/', views.export_sentiment_pdf, name="export_pdf"),
    path('dataset-analysis/', views.dataset_analysis_view, name="dataset_analysis"),
    path('chatbot/', views.chatbot_view, name="chatbot"),
    path('chatbot/api/', views.chatbot_api, name="chatbot_api"),
    path('chatbot/clear/', views.chatbot_clear_api, name="chatbot_clear_api"),
]