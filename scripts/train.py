import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import mlflow

# MLflow experiment tracking
mlflow.start_run()

# Load data from the CSV file
data_path = 'dataset.csv'
data = pd.read_csv(data_path)

# Check if the necessary columns are present
if not all(col in data.columns for col in ['Date', 'Temperature', 'Ice Cream Sales']):
    raise ValueError("The dataset must contain 'Date', 'Temperature', and 'Ice Cream Sales' columns.")

# Prepare the features and target variable
X = data[['Temperature']]
y = data['Ice Cream Sales']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions and calculate mean squared error
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)

# Log parameters and metrics
mlflow.log_param("model_type", "Linear Regression")
mlflow.log_metric("mse", mse)

# Save the model in the outputs folder
output_dir = os.path.join(".", "outputs")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.pkl")
joblib.dump(model, model_path)

mlflow.end_run()
print("Model saved in:", model_path)
