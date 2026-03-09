from django import forms
from .models import BatchUpload


class TweetAnalysisForm(forms.Form):
    tweet_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'placeholder': 'Enter a tweet or any text to analyse for depression signals...',
            'class': 'tweet-input',
        }),
        label='',
        max_length=5000,
        min_length=5,
        error_messages={
            'required':  'Please enter some text to analyse.',
            'min_length': 'Text must be at least 5 characters long.',
        }
    )


class CSVUploadForm(forms.ModelForm):
    class Meta:
        model  = BatchUpload
        fields = ['csv_file']
        widgets = {
            'csv_file': forms.FileInput(attrs={
                'accept': '.csv',
                'class':  'file-input',
            })
        }
        labels = {'csv_file': 'Upload CSV File (must have a "text" column)'}
