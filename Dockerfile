FROM python:3.6-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN echo "backend: Agg" > /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
