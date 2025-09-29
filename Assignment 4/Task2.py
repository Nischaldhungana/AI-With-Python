import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("weight-height.csv")

print("Columns:", df.columns.tolist())

X = df[["Height"]].values
y = df["Weight"].values

plt.figure(figsize=(6,4))
plt.scatter(X, y, s=10, alpha=0.5)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight (scatter)")
plt.tight_layout()
plt.show()


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


plt.figure(figsize=(6,4))
plt.scatter(X, y, s=10, alpha=0.4, label="data")
# for a smooth line, sort by X
idx = np.argsort(X[:,0])
plt.plot(X[idx], y_pred[idx], linewidth=2, label="linear fit")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.title("Linear regression: Weight ~ Height")
plt.tight_layout()
plt.show()


rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Coefficients: intercept = {model.intercept_:.3f}, slope = {model.coef_[0]:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R^2  = {r2:.3f}")


print("\nAssessment:")
print("- If R^2 is close to 1, linear regression explains most variance.")
print("- Check residuals / scatter to see heteroscedasticity or non-linearity.")
