from django.urls import path
from . import views

app_name = 'emotion'

urlpatterns = [
    path('', views.emotion_analysis, name="emotion_anaylsis"),
    path('type/', views.emotion_analysis_type, name="emotion_analysis_type"),
    path('import/', views.emotion_analysis_import, name="emotion_analysis_import"),
    path('export-csv/', views.export_emotion_csv, name="export_csv"),
    path('export-excel/', views.export_emotion_excel, name="export_excel"),
    path('export-pdf/', views.export_emotion_pdf, name="export_pdf"),
]