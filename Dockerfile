FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-simple.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application (using simple service)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "simple_embeddings_service:app"] 
