from django.contrib import admin
from .models import SearchHistory, Notification

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'query', 'analysis_type', 'timestamp')
    list_filter = ('analysis_type', 'timestamp')
    search_fields = ('query', 'user__username')

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('user', 'message', 'is_read', 'timestamp')
    list_filter = ('is_read', 'timestamp')
    search_fields = ('message', 'user__username')
