# Gunakan image Python sebagai base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file proyek ke dalam container
COPY . /app/

# Menentukan perintah yang akan dijalankan saat container dijalankan
CMD ["python", "prometheus_exporter.py"]  

# Expose port yang dibutuhkan untuk Prometheus (misalnya port 5000 untuk exporter)
EXPOSE 5000

# Jika kamu menjalankan API menggunakan Flask atau FastAPI, pastikan kamu menyesuaikan perintahnya

