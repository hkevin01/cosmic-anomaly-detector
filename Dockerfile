# Dockerfile for Cosmic Anomaly Detector
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libhdf5-dev \
    libfftw3-dev \
    wcslib-dev \
    libcfitsio-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY pyproject.toml README.md ./

# Install the package in development mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/output /app/logs /app/temp

# Set up non-root user for security
RUN groupadd -r cosmic && useradd -r -g cosmic cosmic && \
    chown -R cosmic:cosmic /app
USER cosmic

# Expose port for web interface (if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import cosmic_anomaly_detector; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "cosmic_anomaly_detector.cli", "--help"]
