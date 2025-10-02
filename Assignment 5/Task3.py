import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
import numpy as np


df = pd.read_csv("Auto.csv")
df = df.dropna()

X = df.drop(columns=["mpg", "name", "origin"])
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
alphas = [0.1, 0.2, 0.3, 1, 2, 3, 5, 10]
ridge_scores = []
lasso_scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))

    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))


plt.plot(alphas, ridge_scores, label="Ridge")
plt.plot(alphas, lasso_scores, label="LASSO")
plt.xlabel("Alpha")
plt.ylabel("R2 Score")
plt.legend()
plt.show()


"""
Findings:

I used Ridge and LASSO regression to predict car MPG using all numeric features (except mpg, name, and origin).  
Both models were tested with different alpha values.
The best R² score for Ridge was around 0.79 at alpha = 0.1.  
LASSO also worked best at alpha = 0.1 with a similar score.
Smaller alpha values gave better results.  
As alpha increased, model performance dropped due to underfitting.
Ridge performed slightly better, but both models worked well.

"""