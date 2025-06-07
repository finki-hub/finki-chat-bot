#!/bin/bash

set -e

echo "Running database migrations..."
python -m app.migrations

echo "Starting Gunicorn..."
exec gunicorn -c gunicorn.conf.py app.main:app
