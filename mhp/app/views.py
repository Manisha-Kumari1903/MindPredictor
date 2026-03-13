from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import RegisterForm, LoginForm, AnalyseForm
from .models import TweetAnalysis
from .ml_utils import analyse_tweet


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    return render(request, 'app/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password'],
            )
            if user:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password.')
    return render(request, 'app/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    form = AnalyseForm()
    result = None
    if request.method == 'POST':
        form = AnalyseForm(request.POST)
        if form.is_valid():
            raw = form.cleaned_data['tweet_text']
            try:
                data = analyse_tweet(raw)
                obj = TweetAnalysis.objects.create(
                    user=request.user,
                    tweet_text=raw,
                    cleaned_text=data['cleaned_text'],
                    polarity_score=data['polarity_score'],
                    subjectivity=data['subjectivity'],
                    sentiment_label=data['sentiment_label'],
                    result=data['result'],
                    confidence=data['confidence'],
                )
                result = obj
            except Exception as e:
                messages.error(request, f'Analysis error: {str(e)}')

    analyses = TweetAnalysis.objects.filter(user=request.user)
    total    = analyses.count()
    dep      = analyses.filter(result='depressive').count()
    nd       = total - dep
    recent   = analyses.order_by('-analyzed_at')[:6]
    dep_pct  = round(dep / total * 100, 1) if total else 0
    nd_pct   = round(nd  / total * 100, 1) if total else 0

    return render(request, 'app/dashboard.html', {
        'form': form, 'result': result,
        'total': total, 'depressive': dep, 'not_depressive': nd,
        'dep_pct': dep_pct, 'not_dep_pct': nd_pct, 'recent': recent,
    })


@login_required
def analyse_view(request):
    return redirect('dashboard')


@login_required
def depressive_rate(request):
    analyses = TweetAnalysis.objects.filter(user=request.user)
    total    = analyses.count()
    dep      = analyses.filter(result='depressive').count()
    nd       = total - dep
    positive = analyses.filter(sentiment_label='positive').count()
    negative = analyses.filter(sentiment_label='negative').count()
    neutral  = analyses.filter(sentiment_label='neutral').count()
    dep_pct  = round(dep / total * 100, 1) if total else 0
    nd_pct   = round(nd  / total * 100, 1) if total else 0
    dep_list = analyses.filter(result='depressive').order_by('-analyzed_at')

    return render(request, 'app/depressive_rate.html', {
        'total': total, 'depressive': dep, 'not_depressive': nd,
        'positive': positive, 'negative': negative, 'neutral': neutral,
        'dep_pct': dep_pct, 'not_dep_pct': nd_pct,
        'dep_list': dep_list,
    })
