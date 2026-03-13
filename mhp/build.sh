#!/usr/bin/env bash
pip install -r requirements.txt
python -m textblob.download_corpora
python manage.py migrate
python manage.py collectstatic --no-input
