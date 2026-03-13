from django.db import models
from django.contrib.auth.models import User


class TweetAnalysis(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE)
    tweet_text      = models.TextField()
    cleaned_text    = models.TextField(blank=True, default='')
    polarity_score  = models.FloatField(default=0.0)
    subjectivity    = models.FloatField(default=0.0)
    sentiment_label = models.CharField(max_length=20, default='neutral')
    result          = models.CharField(max_length=20, default='not_depressive')
    confidence      = models.FloatField(default=0.0)
    analyzed_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"{self.user.username} — {self.result} ({self.analyzed_at:%Y-%m-%d})"

    @property
    def is_depressive(self):
        return self.result == 'depressive'

    @property
    def polarity_percent(self):
        return round((self.polarity_score + 1) / 2 * 100, 1)
