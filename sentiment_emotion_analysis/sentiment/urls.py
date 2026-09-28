from django.urls import path
from . import views

app_name = 'sentiment'

urlpatterns = [
    path('', views.sentiment_analysis, name="sentiment_anaylsis"),
    path('type/', views.sentiment_analysis_type, name="sentiment_analysis_type"),
    path('import/', views.sentiment_analysis_import, name="sentiment_analysis_import"),
    path('export-pdf/', views.export_sentiment_pdf, name="export_pdf"),
    path('quiz/', views.quiz_view, name="quiz"),
    path('quiz/api/questions/', views.quiz_questions_api, name="quiz_questions_api"),
    path('quiz/api/submit/', views.quiz_submit_answer_api, name="quiz_submit_answer_api"),
    path('playground/', views.playground_view, name="playground"),
    path('playground/api/sample/', views.playground_sample_api, name="playground_sample_api"),
    path('playground/api/analyze/', views.playground_analyze_api, name="playground_analyze_api"),
    path('playground/export-pdf/', views.export_playground_pdf, name="export_playground_pdf"),
    path('playground/export-docx/', views.export_playground_docx, name="export_playground_docx"),
]