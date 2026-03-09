from django.db import models
from django.utils import timezone


class TweetAnalysis(models.Model):
    """Stores individual tweet analysis results."""

    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral',  'Neutral'),
    ]
    RESULT_CHOICES = [
        ('depressive',     'Depressive'),
        ('not_depressive', 'Not Depressive'),
    ]

    tweet_text       = models.TextField()
    cleaned_text     = models.TextField(blank=True)
    polarity_score   = models.FloatField(default=0.0)
    subjectivity     = models.FloatField(default=0.0)
    sentiment_label  = models.CharField(max_length=10, choices=SENTIMENT_CHOICES, default='neutral')
    result           = models.CharField(max_length=20, choices=RESULT_CHOICES, default='not_depressive')
    confidence       = models.FloatField(default=0.0)
    analyzed_at      = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-analyzed_at']
        verbose_name = 'Tweet Analysis'
        verbose_name_plural = 'Tweet Analyses'

    def __str__(self):
        return f"{self.tweet_text[:60]}... → {self.result}"

    @property
    def is_depressive(self):
        return self.result == 'depressive'

    @property
    def polarity_percent(self):
        """Convert polarity from [-1, 1] to [0, 100] for display."""
        return round((self.polarity_score + 1) / 2 * 100, 1)


class BatchUpload(models.Model):
    """Tracks CSV batch upload jobs."""

    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('completed',  'Completed'),
        ('failed',     'Failed'),
    ]

    csv_file         = models.FileField(upload_to='uploads/')
    total_tweets     = models.IntegerField(default=0)
    depressive_count = models.IntegerField(default=0)
    normal_count     = models.IntegerField(default=0)
    accuracy         = models.FloatField(default=0.0)
    status           = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    uploaded_at      = models.DateTimeField(default=timezone.now)
    completed_at     = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Batch #{self.pk} — {self.total_tweets} tweets ({self.status})"

    @property
    def depressive_percent(self):
        if self.total_tweets == 0:
            return 0
        return round(self.depressive_count / self.total_tweets * 100, 1)
