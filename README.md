# Ice Cream Sales Prediction with Azure Machine Learning

Welcome to the Ice Cream Sales Prediction project! This portfolio project showcases how to leverage Azure Machine Learning (Azure ML) to build, track, and deploy a regression model. The model predicts daily ice cream sales based on the ambient temperature, enabling more efficient production planning and waste reduction.
 
---
# 🚀 Project Overview

## Objective

Ice cream sales are closely tied to daily temperatures. The core aim is to use a simple linear regression model to predict ice cream sales from the temperature, thereby giving shop owners the insight they need to optimize production. Along the way, this project illustrates how to:

- Build a synthetic yet realistic dataset,
- Track and experiment with MLflow,
- Manage and deploy models on Azure ML,
- Create an end-to-end machine learning pipeline.

## Real-World Scenario

Imagine you own Gelato Mágico on the coast. Optimizing ice cream production is critical because producing too little results in lost sales while producing too much leads to waste. This project demonstrates how you can harness data—even synthetic data modeled on real trends—to anticipate daily demand.

## 🛠️ Detailed Pipeline Breakdown

## 1. Set Up Azure Machine Learning Workspace

**Purpose:** Establish a central hub where you can manage experiments, compute resources, and model registry.

**Steps:**
- Create an Azure ML Workspace: Either via the [Azure Portal](https://portal.azure.com/) or using the Python SDK.
```python
from azureml.core import Workspace

# Load workspace details from the config.json file
ws = Workspace.from_config()  
print("Workspace loaded:", ws.name)
```
- **Provision a Compute Cluster:** Create a compute cluster that will be used for model training.
```python
from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = "cpu-cluster"
compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_D2_V2', 
    min_nodes=0, 
    max_nodes=4
)

compute_target = ComputeTarget.create(ws, compute_name, compute_config)
compute_target.wait_for_completion(show_output=True)
print("Compute target created:", compute_target.name)
```
This foundational setup will support scalable computation and unified management of your experiments.

## 2. Generate a Synthetic Dataset
**Purpose:** Create a dataset that mimics realistic sales behavior based on ambient temperature.

**What’s Included in the Dataset:**
- **Date:** Generates sequential dates.
- **Temperature (°C):** Values that reflect seasonal variation.
- **Ice Cream Sales (Units Sold):** Derived using a linear trend where higher temperatures yield more sales.

**Download the dataset I used for this project**: [data.csv](https://github.com/user-attachments/files/19726173/data.csv)


**Example Table Format:**
----------------------------------------------------------------------
|       Date       | Temperature (°C) | Ice Cream Sales (Units Sold) |
|------------------|------------------|------------------------------|
| 2025-01-01       |       15         |             120              |
| 2025-01-02       |       17         |             135              |
| 2025-01-03       |       19         |             150              |
| 2025-01-04       |       22         |             180              |
| 2025-01-05       |       24         |             200              |
| ...              |       ...        |             ...              |
| 2025-04-09       |       34         |             380              |
| 2025-04-10       |       32         |             360              |

Generate your 100-row dataset in a pre-processing script and store it in the inputs/ folder. This facilitates reproducibility and easier onboarding of new data later on.
```python
import csv
import random
from datetime import datetime, timedelta
import pandas as pd
from IPython.display import FileLink

# Parameters
num_rows = 100
start_date = datetime.today() - timedelta(days=num_rows)
file_name = "ice_cream_sales_data.csv"

# Generate data
data = []
for i in range(num_rows):
    date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    temp = round(random.uniform(15, 35), 1)
    sales = int(random.gauss(temp * 10 - 50, 20))
    sales = max(sales, 0)
    data.append([date, temp, sales])

# Write to CSV
df = pd.DataFrame(data, columns=["Date", "Temperature (°C)", "Ice Cream Sales (Units Sold)"])
df.to_csv(file_name, index=False)

# Display download link
FileLink(file_name)
```

## 3. Define the Azure ML Environment
**Purpose:** Provide a reproducible setup by specifying dependencies such as Scikit-learn, Pandas, NumPy, MLflow, and others.
**Steps:**
- **Create an Environment:** This encapsulates all Python dependencies.
```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment("ice-cream-env")
deps = CondaDependencies.create(
    pip_packages=[
        "scikit-learn", 
        "pandas", 
        "numpy",
        "mlflow",
        "azureml-sdk"
    ]
)
env.python.conda_dependencies = deps
env.register(ws)
print("Environment registered.")
```
A well-defined environment ensures that your experiments are fully reproducible across different runs and by other team members.

## 4. Train the Regression Model with MLflow Logging
**Purpose:** Train a linear regression model to predict ice cream sales, track your experiment’s parameters, metrics, and artifacts with MLflow, and save the model for later use.

**Steps:**
- **Training Script (train.py):**
  - Load the dataset.
  - Train the model with Scikit-learn’s LinearRegression.
  - Calculate performance metrics (e.g., Mean Squared Error).
  - Log parameters, metrics, and the model artifact using MLflow.
  - Save the model to a designated path.
```python
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import mlflow

# MLflow experiment tracking
mlflow.start_run()

# Simulate synthetic data for 100 rows
# In a real scenario, you could load data from your inputs folder
dates = pd.date_range(start="2025-01-01", periods=100)
temperatures = [15 + 0.2 * i + (i % 5) * 1.5 for i in range(100)]
# Assume the relation: Sales = 100 + 10 * Temperature + noise
ice_cream_sales = [int(100 + 10 * temp + (i % 3) * 5) for i, temp in enumerate(temperatures)]

data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperatures,
    'IceCreamSales': ice_cream_sales
})

X = data[['Temperature']]
y = data['IceCreamSales']

model = LinearRegression()
model.fit(X, y)

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
```
This script forms the core of your experiment and takes advantage of MLflow to maintain a robust record of your model training process.

## 5. Pack Scripts and Management
**Purpose:** Package your training script for execution on the compute cluster managed by Azure ML.

**Submit the Experiment:**
```python
from azureml.core import ScriptRunConfig, Experiment

src = ScriptRunConfig(
    source_directory="./scripts", 
    script="train.py", 
    compute_target=compute_target, 
    environment=env
)

experiment = Experiment(ws, "IceCreamSales")
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
print("Run completed.")
```
This integration brings all the components together, enabling you to track experiments and iterate on your model efficiently.

## 6. Register and Deploy the Model for Real-Time Inference
**Register the Model**
**Purpose:** Capture the trained model version for future deployment and tracking.
```python
from azureml.core.model import Model

model = Model.register(
    workspace=ws,
    model_path='outputs/model.pkl',
    model_name='icecream_sales_regression'
)
print("Model registered:", model.name)
```
**Create a Scoring Script (score.py)**
This script accepts HTTP requests with new temperature data, loads the model, and returns sales predictions.
```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('icecream_sales_regression')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        temperature = data["Temperature"]
        prediction = model.predict(np.array([[temperature]]))
        return json.dumps({"IceCreamSalesPrediction": prediction[0]})
    except Exception as e:
        return json.dumps({"error": str(e)})
```
**Deploy the Model using Azure Container Instances (ACI)**

**Purpose:** Create a deployment configuration for scalable, real-time predictions.
```python
from azureml.core.webservice import AciWebservice, InferenceConfig

inference_config = InferenceConfig(
    entry_script="scripts/score.py", 
    environment=env
)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, 
    memory_gb=1
)

service = Model.deploy(
    ws,
    name="icecream-sales-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
print("Service deployed at:", service.scoring_uri)
```
This step exposes your model as a web service that can be consumed via HTTP requests, completing the pipeline from training to deployment.

---
What I was able to learn completing this project:
✅ Train a Machine Learning model to predict ice cream sales based on the temperature of the day.
✅ Register and manage the model using MLflow.
✅ Implement the model for real-time predictions in a cloud computing environment.
✅ Create a structured pipeline to train and test the model, ensuring reproducibility. 
