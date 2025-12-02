# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# === Load the dataset ===
data = pd.read_csv("AirPassengers.csv")

# Convert 'Month' column to datetime format
data['Month'] = pd.to_datetime(data['Month'], errors='coerce')

# Drop missing or invalid dates
data.dropna(subset=['Month'], inplace=True)

# Sort by date
data = data.sort_values(by='Month')

# Set 'Month' as index
data.set_index('Month', inplace=True)

# Display first few rows
print(data.head())

# === Select the target variable for time series analysis ===
target_col = '#Passengers'

# === Plot the time series ===
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target_col], label=target_col, color='blue')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'{target_col} Time Series')
plt.legend()
plt.grid()
plt.show()

# === Function to check stationarity ===
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# === Check stationarity of the series ===
print("\n--- Stationarity Test for #Passengers ---")
check_stationarity(data[target_col])

# === Plot ACF and PACF ===
plt.figure(figsize=(10, 4))
plot_acf(data[target_col].dropna(), lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(data[target_col].dropna(), lags=30)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# === Train-Test Split ===
train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]

# === Build and fit SARIMA model ===
# The seasonal_order=(1,1,1,12) suits monthly data like AirPassengers
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# === Forecast ===
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# === Evaluate performance ===
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.4f}')

# === Plot predictions vs actuals ===
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'SARIMA Model Predictions for {target_col}')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

<img width="238" height="153" alt="image" src="https://github.com/user-attachments/assets/7ec6133d-f6ca-40f9-b618-5d421bb662d9" />

<img width="926" height="497" alt="image" src="https://github.com/user-attachments/assets/d64f8717-7f9a-4cbc-ba50-4e059fbca933" />

<img width="404" height="161" alt="image" src="https://github.com/user-attachments/assets/3b948709-f744-4fd7-bd02-9383e7dbaa92" />

<img width="722" height="557" alt="image" src="https://github.com/user-attachments/assets/032d0744-10fa-4ab1-8bd0-f06e542ac4a0" />

<img width="703" height="547" alt="image" src="https://github.com/user-attachments/assets/24875882-7f32-4cc3-8fe0-6fff73c1b949" />

<img width="400" height="42" alt="image" src="https://github.com/user-attachments/assets/8f33526b-296d-470d-a001-5fce272a4c7f" />

<img width="926" height="487" alt="image" src="https://github.com/user-attachments/assets/048fee81-98d8-4efe-b286-dffc27f81008" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
