from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count

from .forms import RegisterForm, LoginForm, AnalyseForm
from .models import TweetAnalysis
from .ml_utils import analyse_tweet


# ── REGISTER ─────────────────────────────────────────────────────────────────
def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Account created.')
            return redirect('dashboard')
    return render(request, 'app/register.html', {'form': form})


# ── LOGIN ────────────────────────────────────────────────────────────────────
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'app/login.html', {'form': form})


# ── LOGOUT ───────────────────────────────────────────────────────────────────
def logout_view(request):
    logout(request)
    return redirect('login')


# ── DASHBOARD ────────────────────────────────────────────────────────────────
@login_required
def dashboard(request):
    form   = AnalyseForm()
    result = None

    # Handle analyse form submitted from dashboard
    if request.method == 'POST':
        form = AnalyseForm(request.POST)
        if form.is_valid():
            raw  = form.cleaned_data['tweet_text']
            data = analyse_tweet(raw)
            obj  = TweetAnalysis.objects.create(
                user            = request.user,
                tweet_text      = raw,
                cleaned_text    = data['cleaned_text'],
                polarity_score  = data['polarity_score'],
                subjectivity    = data['subjectivity'],
                sentiment_label = data['sentiment_label'],
                result          = data['result'],
                confidence      = data['confidence'],
            )
            result = obj

    analyses    = TweetAnalysis.objects.filter(user=request.user)
    total       = analyses.count()
    dep         = analyses.filter(result='depressive').count()
    not_dep     = analyses.filter(result='not_depressive').count()
    recent      = analyses.order_by('-analyzed_at')[:6]
    dep_pct     = round(dep / total * 100, 1) if total else 0
    not_dep_pct = round(not_dep / total * 100, 1) if total else 0

    context = {
        'form':           form,
        'result':         result,
        'total':          total,
        'depressive':     dep,
        'not_depressive': not_dep,
        'dep_pct':        dep_pct,
        'not_dep_pct':    not_dep_pct,
        'recent':         recent,
    }
    return render(request, 'app/dashboard.html', context)


# ── ANALYSE ──────────────────────────────────────────────────────────────────
@login_required
def analyse_view(request):
    form   = AnalyseForm()
    result = None

    if request.method == 'POST':
        form = AnalyseForm(request.POST)
        if form.is_valid():
            raw  = form.cleaned_data['tweet_text']
            data = analyse_tweet(raw)
            obj  = TweetAnalysis.objects.create(
                user            = request.user,
                tweet_text      = raw,
                cleaned_text    = data['cleaned_text'],
                polarity_score  = data['polarity_score'],
                subjectivity    = data['subjectivity'],
                sentiment_label = data['sentiment_label'],
                result          = data['result'],
                confidence      = data['confidence'],
            )
            result = obj

    return render(request, 'app/analyse.html', {'form': form, 'result': result})


# ── DEPRESSIVE RATE ──────────────────────────────────────────────────────────
@login_required
def depressive_rate(request):
    analyses = TweetAnalysis.objects.filter(user=request.user)
    total    = analyses.count()
    dep      = analyses.filter(result='depressive').count()
    not_dep  = analyses.filter(result='not_depressive').count()

    dep_pct     = round(dep / total * 100, 1) if total else 0
    not_dep_pct = round(not_dep / total * 100, 1) if total else 0

    positive = analyses.filter(sentiment_label='positive').count()
    negative = analyses.filter(sentiment_label='negative').count()
    neutral  = analyses.filter(sentiment_label='neutral').count()

    # All depressive records
    dep_list = analyses.filter(result='depressive').order_by('-analyzed_at')

    context = {
        'total':          total,
        'depressive':     dep,
        'not_depressive': not_dep,
        'dep_pct':        dep_pct,
        'not_dep_pct':    not_dep_pct,
        'positive':       positive,
        'negative':       negative,
        'neutral':        neutral,
        'dep_list':       dep_list,
    }
    return render(request, 'app/depressive_rate.html', context)
