import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("50_Startups.csv")
print("Sample rows from the dataset:")
print(data.head())
print(data.info())

num_data = data.select_dtypes(include=[np.number])
print("\nCorrelation matrix for numeric features:")
print(num_data.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(num_data.corr().round(2), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)
X = data_encoded.drop('Profit', axis=1)
y = data_encoded['Profit']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(data['R&D Spend'], data['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Profit and R&D Spend')


plt.subplot(1, 2, 2)
plt.scatter(data['Marketing Spend'], data['Profit'], color='green')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Profit and Marketing Spend')


plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred_train = reg_model.predict(X_train)
y_pred_test = reg_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"Training RMSE = {rmse_train:.2f}, R² = {r2_train:.4f}")
print(f"Testing  RMSE = {rmse_test:.2f}, R² = {r2_test:.4f}")

"""
Findings:

- The dataset has 50 rows and includes R&D Spend, Administration, Marketing Spend, State, and Profit.
- R&D Spend has the strongest link to Profit, followed by Marketing Spend.
- Administration doesn’t affect Profit much.
- The scatter plots show a clear trend between Profit and R&D/Marketing Spend.
- The model performs well, with high R² and low RMSE on both train and test data.
- In short, R&D is the biggest factor for predicting profit.

"""
