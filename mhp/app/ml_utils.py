import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'RT\s*:?\s*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def preprocess(text):
    cleaned = clean_text(text)
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    except Exception:
        pass
    try:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    except Exception:
        pass
    return ' '.join(tokens)


def analyse_tweet(raw_text):
    cleaned = preprocess(raw_text)
    try:
        blob = TextBlob(cleaned)
        polarity     = round(blob.sentiment.polarity, 4)
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

    confidence = round(abs(polarity) * 100, 2)

    return {
        'cleaned_text':    cleaned,
        'polarity_score':  polarity,
        'subjectivity':    subjectivity,
        'sentiment_label': sentiment_label,
        'result':          result,
        'confidence':      confidence,
    }
