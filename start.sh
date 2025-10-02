#!/usr/bin/env bash
set -e

# Apply database migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Start Gunicorn server
exec gunicorn detectingbot.wsgi --bind 0.0.0.0:$PORT
