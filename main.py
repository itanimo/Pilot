import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Read the CSV file
data = pd.read_csv('dataset.csv')

# Create a linear regression model and fit it to the data
reg = LinearRegression().fit(data[['income']], data['order_amount'])

# Print the coefficient and intercept of the linear regression line
print(f"Coefficient: {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

# Calculate the average income and order amount
average_income = data['income'].mean()
average_order_amount = data['order_amount'].mean()

# Print the average income and order amount
print(f"Average income: {average_income}")
print(f"Average order amount: {average_order_amount}")

# Create a scatter plot of income vs order amount, with the regression line
plt.scatter(data['income'], data['order_amount'])
plt.plot(data[['income']], reg.predict(data[['income']]), color='red')
plt.xlabel('Income')
plt.ylabel('Order Amount')
plt.title('Income vs Order Amount')
plt.show()
