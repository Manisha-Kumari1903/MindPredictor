# MindScan — Mental Health Prediction using Machine Learning
### Django Web Application | Naive Bayes + NBTree | NLP + Sentiment Analysis

---

## Project Structure

```
mental_health_project/
├── manage.py
├── requirements.txt
├── mental_health_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── depression_app/
    ├── models.py          # TweetAnalysis, BatchUpload models
    ├── views.py           # All page views + AJAX endpoint
    ├── urls.py            # URL routing
    ├── forms.py           # Django forms
    ├── ml_utils.py        # Core ML pipeline (clean → analyse → classify)
    ├── admin.py           # Django admin registration
    ├── ml_model/          # Saved model .pkl files (auto-generated)
    ├── templates/
    │   └── depression_app/
    │       ├── base.html
    │       ├── dashboard.html
    │       ├── analyse.html
    │       ├── batch_upload.html
    │       ├── batch_result.html
    │       ├── history.html
    │       └── train_model.html
    └── static/
        └── depression_app/
```

---

## Setup Instructions

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 3. Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Create superuser (for admin panel)
```bash
python manage.py createsuperuser
```

### 5. Run the development server
```bash
python manage.py runserver
```

### 6. Open in browser
```
http://127.0.0.1:8000/
```

---

## How to Use

1. **Analyse a single tweet** → Go to `/analyse/` → paste tweet → click Analyse
2. **Upload a CSV** → Go to `/batch/` → upload CSV with a `text` column
3. **Train the model** → After analysing ≥ 20 tweets, go to `/train/` → click Train
4. **View history** → Go to `/history/` to see all past analyses
5. **Admin panel** → Go to `/admin/` with superuser credentials

---

## ML Pipeline

```
Raw Tweet
   ↓ Remove URLs, @mentions, #hashtags, retweets
   ↓ Lowercase, strip non-alpha
   ↓ Tokenize (NLTK)
   ↓ Remove stopwords
   ↓ Lemmatize (WordNet)
   ↓ TextBlob sentiment scoring
   ↓    polarity < 0  → depressive
   ↓    polarity >= 0 → not_depressive
   ↓ Naive Bayes (TF-IDF + MultinomialNB) [if model trained]
Result: depressive | not_depressive
```

---

## Algorithms Used

| Algorithm | Description | Accuracy (3k tweets) |
|-----------|-------------|---------------------|
| Naive Bayes | Probabilistic, TF-IDF features | 97.31% |
| NBTree | Hybrid NB + Decision Tree | 97.31% |

---

## Tech Stack

- **Backend**: Python 3, Django 4.2
- **ML**: scikit-learn (MultinomialNB), TextBlob
- **NLP**: NLTK (tokenize, stem, lemmatize)
- **Data**: Pandas, NumPy
- **Database**: SQLite (default)
- **Frontend**: Vanilla HTML/CSS/JS (no frameworks)
