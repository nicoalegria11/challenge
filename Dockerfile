# syntax=docker/dockerfile:1.2
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy your entire application into the container
COPY . .

# Command to run your FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]