"""
ml_utils.py
-----------
Core NLP + ML pipeline for depression detection from tweet text.

Pipeline:
  raw tweet → clean → tokenize → stem/lemmatize
           → sentiment score (TextBlob)
           → label (depressive / not_depressive)
           → Naive Bayes or NBTree classify
"""

import re
import os
import pickle
import logging
from pathlib import Path

import numpy as np

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# ML
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Download NLTK data once ──────────────────────────────────────────────────
_NLTK_DOWNLOADS = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                   'punkt_tab']
for _pkg in _NLTK_DOWNLOADS:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# ── Text Cleaner ─────────────────────────────────────────────────────────────

class TweetCleaner:
    """Cleans raw tweet text for NLP processing."""

    def __init__(self):
        self.stemmer    = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

    def clean(self, text: str) -> str:
        """Full cleaning pipeline: remove noise → lowercase → strip."""
        if not text or not isinstance(text, str):
            return ''
        text = re.sub(r'http\S+|www\S+',        '', text)   # URLs
        text = re.sub(r'@\w+',                   '', text)   # @mentions
        text = re.sub(r'#\w+',                   '', text)   # #hashtags
        text = re.sub(r'RT\s*:?\s*',             '', text)   # retweets
        text = re.sub(r'[^a-zA-Z\s]',            '', text)   # non-alpha
        text = re.sub(r'\s+',                    ' ', text)  # extra spaces
        return text.strip().lower()

    def tokenize(self, text: str) -> list:
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def stem(self, tokens: list) -> list:
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def full_preprocess(self, text: str) -> str:
        """Clean → tokenize → remove stopwords → lemmatize → rejoin."""
        cleaned  = self.clean(text)
        tokens   = self.tokenize(cleaned)
        tokens   = self.remove_stopwords(tokens)
        tokens   = self.lemmatize(tokens)
        return ' '.join(tokens)


# ── Sentiment Analyser ───────────────────────────────────────────────────────

class SentimentAnalyser:
    """
    Uses TextBlob to compute polarity and subjectivity.

    Polarity labels:
        < 0  → negative  → depressive
        = 0  → neutral   → not_depressive
        > 0  → positive  → not_depressive
    """

    @staticmethod
    def analyse(text: str) -> dict:
        try:
            blob = TextBlob(text)
            polarity     = round(blob.sentiment.polarity,     4)
            subjectivity = round(blob.sentiment.subjectivity, 4)
        except Exception:
            polarity, subjectivity = 0.0, 0.0

        if polarity < 0:
            sentiment_label = 'negative'
            result          = 'depressive'
        elif polarity > 0:
            sentiment_label = 'positive'
            result          = 'not_depressive'
        else:
            sentiment_label = 'neutral'
            result          = 'not_depressive'

        return {
            'polarity':        polarity,
            'subjectivity':    subjectivity,
            'sentiment_label': sentiment_label,
            'result':          result,
        }


# ── Model Trainer ────────────────────────────────────────────────────────────

class DepressionClassifier:
    """
    Naive Bayes classifier wrapped in a TF-IDF → MultinomialNB pipeline.
    Mirrors the NBTree approach from the paper by combining feature richness
    with the probabilistic NB classifier.
    """

    def __init__(self, model_path=None, vectorizer_path=None):
        self.model_path      = model_path
        self.vectorizer_path = vectorizer_path
        self.pipeline        = None
        self.is_trained      = False
        self._load_model()

    def _load_model(self):
        """Load pre-trained model if files exist."""
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                self.is_trained = True
                logger.info("Model loaded from %s", self.model_path)
            except Exception as e:
                logger.warning("Could not load model: %s", e)

    def build_pipeline(self):
        """Create a TF-IDF + Naive Bayes pipeline."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=2,
            )),
            ('clf', MultinomialNB(alpha=0.5)),
        ])
        return self.pipeline

    def train(self, texts: list, labels: list) -> dict:
        """
        Train the model and return evaluation metrics.

        Args:
            texts:  list of cleaned tweet strings
            labels: list of 'depressive' / 'not_depressive'

        Returns:
            dict with accuracy, precision, recall, f1, confusion_matrix
        """
        if len(texts) < 10:
            return {'error': 'Need at least 10 samples to train.'}

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.30, random_state=42, stratify=labels
        )

        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)

        metrics = {
            'accuracy':         round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision':        round(precision_score(y_test, y_pred, pos_label='depressive', zero_division=0) * 100, 2),
            'recall':           round(recall_score(y_test, y_pred,    pos_label='depressive', zero_division=0) * 100, 2),
            'f1_score':         round(f1_score(y_test, y_pred,        pos_label='depressive', zero_division=0) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['depressive', 'not_depressive']).tolist(),
            'train_size':       len(X_train),
            'test_size':        len(X_test),
            'report':           classification_report(y_test, y_pred),
        }

        self._save_model()
        logger.info("Model trained. Accuracy: %.2f%%", metrics['accuracy'])
        return metrics

    def predict(self, text: str) -> dict:
        """Predict depression label for a single cleaned tweet."""
        if not self.is_trained or self.pipeline is None:
            return {'result': 'not_depressive', 'confidence': 0.0, 'error': 'Model not trained'}
        try:
            proba  = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_
            idx    = list(classes).index('depressive') if 'depressive' in classes else 0
            result = self.pipeline.predict([text])[0]
            return {
                'result':     result,
                'confidence': round(float(proba[idx]) * 100, 2),
            }
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return {'result': 'not_depressive', 'confidence': 0.0}

    def _save_model(self):
        if self.model_path:
            os.makedirs(Path(self.model_path).parent, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            logger.info("Model saved to %s", self.model_path)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def analyse_tweet(raw_text: str, classifier: DepressionClassifier = None) -> dict:
    """
    Full pipeline for a single tweet.

    Returns a dict suitable for saving to TweetAnalysis model.
    """
    cleaner   = TweetCleaner()
    analyser  = SentimentAnalyser()

    cleaned   = cleaner.full_preprocess(raw_text)
    sentiment = analyser.analyse(cleaned)

    result = {
        'cleaned_text':    cleaned,
        'polarity_score':  sentiment['polarity'],
        'subjectivity':    sentiment['subjectivity'],
        'sentiment_label': sentiment['sentiment_label'],
        'result':          sentiment['result'],
        'confidence':      0.0,
    }

    # If a trained ML model is available, use it (overrides sentiment-only result)
    if classifier and classifier.is_trained and cleaned:
        pred = classifier.predict(cleaned)
        result['result']     = pred.get('result', result['result'])
        result['confidence'] = pred.get('confidence', 0.0)

    return result
