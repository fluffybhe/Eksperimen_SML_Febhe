import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(raw_data):
    # Mengisi missing values dengan rata-rata untuk kolom total_bedrooms
    raw_data['total_bedrooms'].fillna(raw_data['total_bedrooms'].mean(), inplace=True)
    
    # Label encoding untuk kolom 'ocean_proximity'
    label_encoder = LabelEncoder()
    raw_data['ocean_proximity'] = label_encoder.fit_transform(raw_data['ocean_proximity'])
    
    # Normalisasi data numerik
    scaler = StandardScaler()
    features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    raw_data[features] = scaler.fit_transform(raw_data[features])
    
    return raw_data

# Membaca dataset mentah
raw_data = pd.read_csv('../namadataset_raw/california_housing.csv')

# Melakukan preprocessing
processed_data = preprocess_data(raw_data)

# Menyimpan dataset yang telah diproses
processed_data.to_csv('../namadataset_preprocessing/processed_california_housing.csv', index=False)

print("Preprocessing complete and data saved.")


