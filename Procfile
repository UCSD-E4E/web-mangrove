web: gunicorn app:server --timeout 200 --preload 
worker: celery worker -A app.celery --loglevel=info