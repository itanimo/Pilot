import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

mean_order_amount = data['order_amount'].mean()
median_order_amount = data['order_amount'].median()
mode_order_amount = data['order_amount'].mode()

plt.hist(data['order_amount'])
plt.xlabel('Order Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Order Amount')
plt.show()
