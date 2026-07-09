FROM python:3.10-slim

WORKDIR /app

# Copy requirements from Major Project
COPY "Major Project/requirements.txt" .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default port and run command
ENV PORT=7860
EXPOSE 7860

# Run the app
CMD ["python", "Major Project/src/app.py"]
