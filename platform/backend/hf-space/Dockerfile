FROM python:3.11-slim

WORKDIR /app
ENV APP_ENV=production DATABASE_PATH=/data/app.db CURRICULUM_DIR=/app/curriculum

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /data

EXPOSE 7860

CMD ["python", "scripts/start_persistent.py"]
