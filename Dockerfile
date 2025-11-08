FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Add timeout + retry
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY . .

CMD ["python", "src/app.py"]
	