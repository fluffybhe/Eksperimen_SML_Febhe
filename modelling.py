import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Membaca data yang telah diproses
data = pd.read_csv('namadataset_preprocessing/processed_california_housing.csv')

# Memisahkan fitur dan target
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Membagi data menjadi data latih dan data uji (80% untuk latih, 20% untuk uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Melakukan pelatihan dengan MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)

    # Logging model ke MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Evaluasi model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Logging metrik ke MLflow
    mlflow.log_metric("mae", mae)
    print(f'Mean Absolute Error: {mae}')
