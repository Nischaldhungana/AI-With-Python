import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("weight-height.csv")

plt.scatter(data['Height'], data['Weight'])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("scatter plot: Height & Weight")
plt.show()

X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, color='purple')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Regression Line: Height & Weight")
plt.show()

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")



