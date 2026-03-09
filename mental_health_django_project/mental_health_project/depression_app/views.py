import io
import csv
import json
import logging
from datetime import datetime

import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from django.utils import timezone
from django.db.models import Count
from django.core.paginator import Paginator

from .models import TweetAnalysis, BatchUpload
from .forms import TweetAnalysisForm, CSVUploadForm
from .ml_utils import analyse_tweet, TweetCleaner, DepressionClassifier

logger = logging.getLogger(__name__)

# ── Singleton classifier ─────────────────────────────────────────────────────
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = DepressionClassifier(
            model_path=settings.MODEL_PATH,
            vectorizer_path=settings.VECTORIZER_PATH,
        )
    return _classifier


# ── Dashboard ────────────────────────────────────────────────────────────────

def dashboard(request):
    """Home page — analytics overview."""
    total    = TweetAnalysis.objects.count()
    dep      = TweetAnalysis.objects.filter(result='depressive').count()
    not_dep  = TweetAnalysis.objects.filter(result='not_depressive').count()
    batches  = BatchUpload.objects.count()
    recent   = TweetAnalysis.objects.order_by('-analyzed_at')[:8]

    dep_pct     = round(dep / total * 100, 1)     if total else 0
    not_dep_pct = round(not_dep / total * 100, 1) if total else 0

    # Sentiment distribution for chart
    sentiment_data = {
        'positive': TweetAnalysis.objects.filter(sentiment_label='positive').count(),
        'negative': TweetAnalysis.objects.filter(sentiment_label='negative').count(),
        'neutral':  TweetAnalysis.objects.filter(sentiment_label='neutral').count(),
    }

    # Last 7 analyses dates for mini chart
    last_7 = (
        TweetAnalysis.objects
        .values('analyzed_at__date')
        .annotate(count=Count('id'))
        .order_by('-analyzed_at__date')[:7]
    )

    context = {
        'total':          total,
        'depressive':     dep,
        'not_depressive': not_dep,
        'dep_pct':        dep_pct,
        'not_dep_pct':    not_dep_pct,
        'batches':        batches,
        'recent':         recent,
        'sentiment_json': json.dumps(sentiment_data),
        'model_trained':  get_classifier().is_trained,
        'last_7':         list(last_7),
    }
    return render(request, 'depression_app/dashboard.html', context)


# ── Single Tweet Analysis ────────────────────────────────────────────────────

def analyse_view(request):
    """Single tweet analysis page."""
    form   = TweetAnalysisForm()
    result = None

    if request.method == 'POST':
        form = TweetAnalysisForm(request.POST)
        if form.is_valid():
            raw_text   = form.cleaned_data['tweet_text']
            classifier = get_classifier()

            analysis_data = analyse_tweet(raw_text, classifier)

            # Save to DB
            obj = TweetAnalysis.objects.create(
                tweet_text      = raw_text,
                cleaned_text    = analysis_data['cleaned_text'],
                polarity_score  = analysis_data['polarity_score'],
                subjectivity    = analysis_data['subjectivity'],
                sentiment_label = analysis_data['sentiment_label'],
                result          = analysis_data['result'],
                confidence      = analysis_data['confidence'],
            )
            result = obj

    context = {'form': form, 'result': result}
    return render(request, 'depression_app/analyse.html', context)


# ── AJAX single predict ──────────────────────────────────────────────────────

@require_POST
def predict_ajax(request):
    """AJAX endpoint for live prediction as user types."""
    text = request.POST.get('text', '').strip()
    if not text:
        return JsonResponse({'error': 'No text provided'}, status=400)

    try:
        classifier    = get_classifier()
        analysis_data = analyse_tweet(text, classifier)
        return JsonResponse({
            'result':          analysis_data['result'],
            'confidence':      analysis_data['confidence'],
            'polarity':        analysis_data['polarity_score'],
            'subjectivity':    analysis_data['subjectivity'],
            'sentiment_label': analysis_data['sentiment_label'],
        })
    except Exception as e:
        logger.error("AJAX predict error: %s", e)
        return JsonResponse({'error': str(e)}, status=500)


# ── Batch CSV Upload ─────────────────────────────────────────────────────────

def batch_upload(request):
    """Upload a CSV of tweets for bulk analysis."""
    form = CSVUploadForm()

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            batch = form.save(commit=False)
            batch.status = 'processing'
            batch.save()

            try:
                csv_file = request.FILES['csv_file']
                decoded  = csv_file.read().decode('utf-8', errors='ignore')
                reader   = csv.DictReader(io.StringIO(decoded))

                if 'text' not in (reader.fieldnames or []):
                    messages.error(request, 'CSV must have a "text" column.')
                    batch.delete()
                    return render(request, 'depression_app/batch_upload.html', {'form': form})

                classifier = get_classifier()
                records    = []
                dep_count  = 0

                for row in reader:
                    raw = row.get('text', '').strip()
                    if not raw:
                        continue
                    data = analyse_tweet(raw, classifier)
                    records.append(TweetAnalysis(
                        tweet_text      = raw,
                        cleaned_text    = data['cleaned_text'],
                        polarity_score  = data['polarity_score'],
                        subjectivity    = data['subjectivity'],
                        sentiment_label = data['sentiment_label'],
                        result          = data['result'],
                        confidence      = data['confidence'],
                    ))
                    if data['result'] == 'depressive':
                        dep_count += 1

                TweetAnalysis.objects.bulk_create(records)

                batch.total_tweets     = len(records)
                batch.depressive_count = dep_count
                batch.normal_count     = len(records) - dep_count
                batch.status           = 'completed'
                batch.completed_at     = timezone.now()
                batch.save()

                messages.success(
                    request,
                    f'Batch complete! Analysed {len(records)} tweets. '
                    f'{dep_count} depressive ({batch.depressive_percent}%).'
                )
                return redirect('batch_result', pk=batch.pk)

            except Exception as e:
                logger.error("Batch upload error: %s", e)
                batch.status = 'failed'
                batch.save()
                messages.error(request, f'Processing failed: {str(e)}')

    batches = BatchUpload.objects.all()[:10]
    context = {'form': form, 'batches': batches}
    return render(request, 'depression_app/batch_upload.html', context)


def batch_result(request, pk):
    """Results for a completed batch upload."""
    batch    = get_object_or_404(BatchUpload, pk=pk)
    analyses = TweetAnalysis.objects.order_by('-analyzed_at')[:batch.total_tweets]
    paginator = Paginator(analyses, 20)
    page      = paginator.get_page(request.GET.get('page'))

    sentiment_data = {
        'positive': sum(1 for a in analyses if a.sentiment_label == 'positive'),
        'negative': sum(1 for a in analyses if a.sentiment_label == 'negative'),
        'neutral':  sum(1 for a in analyses if a.sentiment_label == 'neutral'),
    }

    context = {
        'batch':          batch,
        'analyses':       page,
        'sentiment_json': json.dumps(sentiment_data),
    }
    return render(request, 'depression_app/batch_result.html', context)


# ── History ──────────────────────────────────────────────────────────────────

def history(request):
    """Paginated list of all analysed tweets."""
    qs        = TweetAnalysis.objects.all()
    filter_by = request.GET.get('filter', 'all')

    if filter_by == 'depressive':
        qs = qs.filter(result='depressive')
    elif filter_by == 'not_depressive':
        qs = qs.filter(result='not_depressive')

    paginator = Paginator(qs, 15)
    page      = paginator.get_page(request.GET.get('page'))

    context = {'analyses': page, 'filter_by': filter_by, 'total': qs.count()}
    return render(request, 'depression_app/history.html', context)


# ── Train Model ──────────────────────────────────────────────────────────────

def train_model(request):
    """Train/retrain the ML model from DB data."""
    context   = {'metrics': None, 'error': None}
    queryset  = TweetAnalysis.objects.all()
    total     = queryset.count()
    context['total'] = total

    if request.method == 'POST':
        if total < 20:
            context['error'] = f'Need at least 20 analysed tweets to train. You have {total}.'
            return render(request, 'depression_app/train_model.html', context)

        texts  = list(queryset.values_list('cleaned_text', flat=True))
        labels = list(queryset.values_list('result', flat=True))

        classifier = get_classifier()
        metrics    = classifier.train(texts, labels)

        if 'error' in metrics:
            context['error'] = metrics['error']
        else:
            context['metrics'] = metrics
            messages.success(request, f"Model trained! Accuracy: {metrics['accuracy']}%")

    return render(request, 'depression_app/train_model.html', context)
