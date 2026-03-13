from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class TweetAnalysis(models.Model):
    RESULT_CHOICES = [
        ('depressive',     'Depressive'),
        ('not_depressive', 'Not Depressive'),
    ]
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral',  'Neutral'),
    ]

    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    tweet_text      = models.TextField()
    cleaned_text    = models.TextField(blank=True)
    polarity_score  = models.FloatField(default=0.0)
    subjectivity    = models.FloatField(default=0.0)
    sentiment_label = models.CharField(max_length=10, choices=SENTIMENT_CHOICES, default='neutral')
    result          = models.CharField(max_length=20, choices=RESULT_CHOICES, default='not_depressive')
    confidence      = models.FloatField(default=0.0)
    analyzed_at     = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"{self.tweet_text[:50]} → {self.result}"

    @property
    def is_depressive(self):
        return self.result == 'depressive'

    @property
    def polarity_percent(self):
        return round((self.polarity_score + 1) / 2 * 100, 1)
