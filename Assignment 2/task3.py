import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('weight-height.csv')

length = data['Height'].values
weight = data['Weight'].values

length = length * 2.54
weight = weight * 0.453592

mean_length = np.mean(length)
mean_weight = np.mean(weight)

print("Mean Length in cm:", mean_length)
print("Mean Weight in kg:", mean_weight)

plt.hist(length, bins=20, color='gray', edgecolor='black')
plt.title('Histogram of Lengths')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.show()
