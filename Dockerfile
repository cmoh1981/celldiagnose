# Dockerfile for CellDiagnose-AI Streamlit App
# Hugging Face Spaces Docker deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user

# Set environment variables for Streamlit
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Ensure .streamlit config is accessible
RUN mkdir -p /home/user/.streamlit && \
    cp -r .streamlit/* /home/user/.streamlit/ 2>/dev/null || true && \
    chown -R user:user /home/user/.streamlit 2>/dev/null || true

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
