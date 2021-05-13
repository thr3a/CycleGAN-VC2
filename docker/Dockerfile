FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y libsndfile1-dev
RUN pip install -r requirements.txt
RUN pip install matplotlib
