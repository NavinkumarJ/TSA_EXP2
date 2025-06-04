### DEVELOPED BY: NAVIN KUMAR J
### REGISTER NO: 212222240071
### DATE:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

## AIM:
To Implement Linear and Polynomial Trend Estimation Using Python.

## ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

## PROGRAM:
```python
A - LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('future_gold_price.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

numeric_columns = ['Open', 'High', 'Low', 'Close']
for col in numeric_columns:
    data[col] = data[col].replace(',', '', regex=True).astype(float)

data = data.sort_values('Date')

data['Days'] = (data['Date'] - data['Date'].min()).dt.days

X = data[['Days']]  # Days is the independent variable (feature)
y = data['Close']   # Close price is the dependent variable (target)

model = LinearRegression()
model.fit(X, y)

data['Trend'] = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Actual Close Price', color='blue')
plt.plot(data['Date'], data['Trend'], label='Linear Trend', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Furure Gold Price Linear Trend Estimation')
plt.legend()
plt.show()

print(f"Slope (Rate of Change): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")




B- POLYNOMIAL TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('future_gold_price.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

numeric_columns = ['Open', 'High', 'Low', 'Close']
for col in numeric_columns:
    data[col] = data[col].replace(',', '', regex=True).astype(float)

data = data.sort_values('Date')


data['Days'] = (data['Date'] - data['Date'].min()).dt.days

X = data[['Days']]  # Days is the independent variable (feature)
y = data['Close']   # Close price is the dependent variable (target)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

data['Poly_Trend'] = model.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Actual Close Price', color='blue')
plt.plot(data['Date'], data['Poly_Trend'], label=f'Polynomial Trend (Degree 2)', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(' Future Gold Price Polynomial Trend Estimation')
plt.legend()
plt.show()

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

```
## OUTPUT:
A - LINEAR TREND ESTIMATION

![Screenshot 2024-09-20 084116](https://github.com/user-attachments/assets/99081810-ba81-4a3a-989f-40fe13ea5627)


B- POLYNOMIAL TREND ESTIMATION
![Screenshot 2024-09-20 084130](https://github.com/user-attachments/assets/2afb99ec-0037-4bea-a3cf-a00a37ab53fe)



## RESULT:
Thus the python program for linear and Polynomial Trend Estimation has been executed successfully.
