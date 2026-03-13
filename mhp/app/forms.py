from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model  = User
        fields = ['username', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-input'})


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-input'})


class AnalyseForm(forms.Form):
    tweet_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class':       'form-input',
            'rows':        4,
            'placeholder': 'Enter a tweet or any text to analyse...',
        }),
        label='',
        min_length=5,
        max_length=5000,
    )
