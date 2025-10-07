FROM python:3.10-slim

WORKDIR /hardhat

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 7002

CMD ["python", "app.py"]
