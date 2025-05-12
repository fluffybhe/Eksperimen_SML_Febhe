from prometheus_client import start_http_server, Summary
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Membuat metrik untuk memonitor waktu yang dibutuhkan untuk proses preprocessing
PREPROCESS_TIME = Summary('data_preprocessing_duration_seconds', 'Time spent preprocessing the data')

@PREPROCESS_TIME.time()
def preprocess_data(raw_data):
    # Mengisi missing values dengan rata-rata untuk kolom total_bedrooms
    raw_data['total_bedrooms'] = raw_data['total_bedrooms'].fillna(raw_data['total_bedrooms'].mean())

    # Label encoding untuk kolom 'ocean_proximity'
    label_encoder = LabelEncoder()
    raw_data['ocean_proximity'] = label_encoder.fit_transform(raw_data['ocean_proximity'])
    
    # Normalisasi data numerik
    scaler = StandardScaler()
    features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    raw_data[features] = scaler.fit_transform(raw_data[features])
    
    return raw_data

def read_data():
    # Membaca dataset mentah
    raw_data = pd.read_csv('namadataset_raw/california_housing.csv')
    return raw_data

def save_processed_data(processed_data):
    # Menyimpan dataset yang telah diproses
    processed_data.to_csv('namadataset_preprocessing/processed_california_housing.csv', index=False)
    print("Preprocessing complete and data saved.")

if __name__ == '__main__':
    # Mulai HTTP server pada port 5000
    start_http_server(5000)  # Port yang digunakan untuk Prometheus mengambil metrik
    print("Prometheus Exporter berjalan di port 5000")
    
    while True:
        # Membaca data
        raw_data = read_data()
        
        # Melakukan preprocessing
        processed_data = preprocess_data(raw_data)
        
        # Menyimpan data yang telah diproses
        save_processed_data(processed_data)
        
        # Delay antara pemanggilan fungsi (sesuaikan dengan kebutuhanmu)
        time.sleep(1)  # Interval antara eksekusi (bisa diatur lebih besar jika perlu)
