import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import mlflow
import mlflow.sklearn
import dagshub

# Inisialisasi koneksi ke DagsHub
dagshub.init(repo_owner='fluffybhe', repo_name='Eksperimen_SML_Febhe', mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow")

# Set experiment name (opsional tapi disarankan)
mlflow.set_experiment("california_housing_experiment")

# Path dataset
data_path = "namadataset_preprocessing/processed_california_housing.csv"

# Load dataset
data = pd.read_csv(data_path)

# Pisahkan fitur dan target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Aktifkan autologging (optional, logging otomatis model dan param)
mlflow.sklearn.autolog()

# Mulai training model
with mlflow.start_run(run_name="random_forest_basic"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Logging metrik manual (untuk memastikan eksplisit)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("explained_variance_score", evs)

    # Logging parameter dataset
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
