FROM python:3.10-slim

WORKDIR /smoking

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 7001

CMD ["python", "app.py"]
