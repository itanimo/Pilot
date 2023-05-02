import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('dataset.csv')

# Calculate the average income and order amount
average_income = data['income'].mean()
average_order_amount = data['order_amount'].mean()

# Print the average income and order amount
print(f"Average income: {average_income}")
print(f"Average order amount: {average_order_amount}")

# Create a scatter plot of income vs order amount
plt.scatter(data['income'], data['order_amount'])
plt.xlabel('Income')
plt.ylabel('Order Amount')
plt.title('Income vs Order Amount')
plt.show()
