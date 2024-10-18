
### Name: Thiyagarajan A
### Reg.no: 212222240110
### Date: 
# Ex.No: 07    AUTO REGRESSIVE MODEL
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load and clean the CSV file
data = pd.read_csv("Google_Stock_Price_Train.csv")  # Adjust path if necessary
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data['Close'] = pd.to_numeric(data['Close'].str.replace(',', ''), errors='coerce')  # Clean 'Close' prices
data['Volume'] = pd.to_numeric(data['Volume'].str.replace(',', ''), errors='coerce')  # Clean 'Volume'

# Set 'Date' as index
data.set_index('Date', inplace=True)

# Perform Augmented Dickey-Fuller (ADF) test for stationarity
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Fit an AutoRegressive (AR) model with 13 lags on the training data
lag_order = 13
model = AutoReg(train_data['Close'], lags=lag_order)
model_fit = model.fit()

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plot_acf(data['Close'])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Close'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Compare the predictions with the test data
mse = mean_squared_error(test_data['Close'], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot the test data and predictions
plt.plot(test_data.index, test_data['Close'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price (Close)')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.show()
```
### OUTPUT:

#### GIVEN DATA
![image](https://github.com/user-attachments/assets/04c0476d-2375-46b2-aa97-ff5f1547f084)

#### PACF - ACF
![image](https://github.com/user-attachments/assets/b908bb9e-1b6e-44af-93bb-24cb2d7a521a)
![image](https://github.com/user-attachments/assets/41b128d4-e7f2-49e8-a49d-d8a24613bfd7)


#### PREDICTION
![image](https://github.com/user-attachments/assets/cc5af79e-1147-4768-a422-bbe07813de81)


#### FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/446b4f7b-2a5d-45e0-b079-3dddb632faf0)

### RESULT:
Thus the implementation of auto regression function using python was successfully completed.
