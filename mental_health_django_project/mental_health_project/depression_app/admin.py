from django.contrib import admin
from .models import TweetAnalysis, BatchUpload


@admin.register(TweetAnalysis)
class TweetAnalysisAdmin(admin.ModelAdmin):
    list_display  = ('tweet_text_short', 'result', 'sentiment_label',
                     'polarity_score', 'confidence', 'analyzed_at')
    list_filter   = ('result', 'sentiment_label')
    search_fields = ('tweet_text',)
    readonly_fields = ('cleaned_text', 'polarity_score', 'subjectivity',
                       'sentiment_label', 'result', 'confidence', 'analyzed_at')

    def tweet_text_short(self, obj):
        return obj.tweet_text[:80] + ('...' if len(obj.tweet_text) > 80 else '')
    tweet_text_short.short_description = 'Tweet'


@admin.register(BatchUpload)
class BatchUploadAdmin(admin.ModelAdmin):
    list_display  = ('pk', 'total_tweets', 'depressive_count',
                     'normal_count', 'status', 'uploaded_at')
    list_filter   = ('status',)
    readonly_fields = ('total_tweets', 'depressive_count', 'normal_count',
                       'status', 'uploaded_at', 'completed_at')
