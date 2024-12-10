# Updated model training script
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
data_path = "data/train.csv"
traffic_data = pd.read_csv(data_path)

# Preprocess data
traffic_data['date_time'] = pd.to_datetime(traffic_data['date_time'])
traffic_data['hour'] = traffic_data['date_time'].dt.hour
traffic_data['day_of_week'] = traffic_data['date_time'].dt.dayofweek  # Adding day of the week

# Log transform the target variable (traffic volume) to reduce variance
traffic_data['log_traffic_volume'] = traffic_data['traffic_volume'].apply(lambda x: np.log(x + 1))

# Features and target (removed temperature)
X = traffic_data[['hour', 'clouds_all', 'day_of_week']]
y = traffic_data['log_traffic_volume']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model and scaler
joblib.dump(model, "traffic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and Scaler saved.")
