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
    path('quiz/', views.quiz_view, name="quiz"),
    path('quiz/api/questions/', views.quiz_questions_api, name="quiz_questions_api"),
    path('quiz/api/submit/', views.quiz_submit_answer_api, name="quiz_submit_answer_api"),
    path('playground/', views.playground_view, name="playground"),
    path('playground/api/sample/', views.playground_sample_api, name="playground_sample_api"),
    path('playground/api/analyze/', views.playground_analyze_api, name="playground_analyze_api"),
    path('playground/export-pdf/', views.export_playground_pdf, name="export_playground_pdf"),
    path('playground/export-docx/', views.export_playground_docx, name="export_playground_docx"),
]