import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data['frame']

plt.hist(df['target'], bins=25, color='orange')
plt.xlabel('Disease Score')
plt.ylabel('Frequency')
plt.title('Target Distribution')
plt.show()

sns.heatmap(df.corr().round(2), annot=True)
plt.show()


plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'], color='purple', alpha=0.7)
plt.xlabel('BMI Value')
plt.ylabel('Target')
plt.title('BMI and Target')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'], color='teal', alpha=0.7)
plt.xlabel('s5 Value')
plt.ylabel('Target')
plt.title('s5 and Target')
plt.grid(True)
plt.tight_layout()
plt.show()

X_base = df[['bmi', 's5']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=5)
model_base = LinearRegression()
model_base.fit(X_train, y_train)
y_pred_train_base = model_base.predict(X_train)
y_pred_test_base = model_base.predict(X_test)


rmse_train_base = np.sqrt(mean_squared_error(y_train, y_pred_train_base))
r2_train_base = r2_score(y_train, y_pred_train_base)

rmse_test_base = np.sqrt(mean_squared_error(y_test, y_pred_test_base))
r2_test_base = r2_score(y_test, y_pred_test_base)

print("Model with: bmi + s5")
print(f"Train RMSE = {rmse_train_base:.2f}, R² = {r2_train_base:.4f}")
print(f"Test  RMSE = {rmse_test_base:.2f}, R² = {r2_test_base:.4f}")

X_extended = df[['bmi', 's5', 'bp']]
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y, test_size=0.2, random_state=5)
model_ext = LinearRegression()
model_ext.fit(X_train_ext, y_train_ext)
y_pred_train_ext = model_ext.predict(X_train_ext)
y_pred_test_ext = model_ext.predict(X_test_ext)

rmse_train_ext = np.sqrt(mean_squared_error(y_train_ext, y_pred_train_ext))
r2_train_ext = r2_score(y_train_ext, y_pred_train_ext)
rmse_test_ext = np.sqrt(mean_squared_error(y_test_ext, y_pred_test_ext))
r2_test_ext = r2_score(y_test_ext, y_pred_test_ext)

print("\nModel with: bmi + s5 + bp")
print(f"Train RMSE = {rmse_train_ext:.2f}, R² = {r2_train_ext:.4f}")
print(f"Test  RMSE = {rmse_test_ext:.2f}, R² = {r2_test_ext:.4f}")



"""
a. Which variable would you add next? Why? 

I would add the variable bp (blood pressure) next
because it has a good correlation with the target, 
just like bmi and s5. When I checked the heatmap,
bp stood out as one of the stronger features. 
It makes sense too, since blood pressure is
often related to health and diabetes.

b. How does adding it affect the model's performance?

After adding bp, the model's performance improved. 
The RMSE got smaller, which means the model made fewer
errors, and the R² score increased, which shows it 
explained the data better. So overall, adding bp 
made the model more accurate compared to using only bmi and s5.

c. Does it help if you add even more variables?

Yes, adding more useful variables can help, but only to a point. 
If the extra variables are related to the target, they can 
improve the model. But if they don’t have much connection, 
it might not help and could even make the model worse by 
adding noise. So it's better to choose the important ones,
not just add everything.

"""