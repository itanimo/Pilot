import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load the data into a pandas dataframe
df = pd.read_csv('dataset.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['income']], df['order_amount'], test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the order amount for the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

""" 
The Mean Squared Error (MSE) measures the average of the squared differences between the predicted and actual values. It is calculated as:
MSE = 1/n * ∑(yi - ŷi)^2
where n is the number of observations, yi is the actual value of the i-th observation, and ŷi is the predicted value of the i-th observation.
The Root Mean Squared Error (RMSE) is simply the square root of the MSE, which gives the same unit of measurement as the target variable. It is calculated as:
RMSE = √(MSE)
Both MSE and RMSE are useful for comparing the performance of different regression models. 
A lower MSE or RMSE indicates a better fit of the model to the data. 

"""

print('R-squared:', r_squared)
print('MSE:', mse)
print('RMSE:', rmse)

# Visualize the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Income')
plt.ylabel('Order Amount')
plt.show()
