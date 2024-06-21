FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk punkt data during build
RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
