FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy application code
COPY src/ src/
COPY data/processed/ data/processed/
COPY models/ models/
COPY scripts/ scripts/

# Create directories
RUN mkdir -p reports logs

# Expose port for API
EXPOSE 8000
EXPOSE 8501

# Run the API by default
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]