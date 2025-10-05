# Use a smaller, official Python base image
FROM python:3.11-slim

WORKDIR /Python ./

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install venv module and pip upgrade
RUN apt-get update && apt-get install -y python3-venv \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Make sure all commands use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker cache
COPY Python/requirements.txt .

# Install dependencies into the venv
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY Python/ .

# Expose the Flask app port
EXPOSE 8000

# Default command
CMD ["python", "app.py"]
