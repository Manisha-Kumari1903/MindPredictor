#!/usr/bin/env bash
set -e
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab averaged_perceptron_tagger
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --no-input
