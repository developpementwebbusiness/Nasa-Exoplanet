    # Use a smaller, official Python base image
    FROM python:3.13-slim
    WORKDIR /app

    # Prevent Python from writing .pyc files and enable unbuffered output
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1

    # Copy only requirements first to leverage Docker cache
    COPY Python/requirements.txt .
    # Install dependencies into the venv
    RUN pip install --no-cache-dir -r requirements.txt

    # Expose the Flask app port
    EXPOSE 8000
    WORKDIR /app/Python/server/
    # Default command
    CMD ["python", "app.py"]
