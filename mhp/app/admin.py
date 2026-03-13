from django.contrib import admin
from .models import TweetAnalysis

@admin.register(TweetAnalysis)
class TweetAnalysisAdmin(admin.ModelAdmin):
    list_display  = ('tweet_text', 'result', 'sentiment_label', 'polarity_score', 'analyzed_at')
    list_filter   = ('result', 'sentiment_label')
    search_fields = ('tweet_text',)
