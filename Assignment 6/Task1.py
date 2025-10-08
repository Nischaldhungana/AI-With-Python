import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('bank.csv', delimiter=';')
print("Initial Data Overview:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nShape of Dataset:", df.shape)
print("\nColumn Names:\n", list(df.columns))

selected_cols = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']
df2 = df[selected_cols].copy()

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

if df3['y'].dtype == 'object':
    df3['y'] = df3['y'].map({'no': 0, 'yes': 1})

corr_matrix = df3.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of df3")
plt.tight_layout()
plt.show()

print("\nTop features most correlated with target (y):\n")
top_corr = corr_matrix['y'].drop('y').abs().sort_values(ascending=False)
print(top_corr.head(10))

X = df3.drop('y', axis=1)
y = df3['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

metrics.ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test, cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\nK-Nearest Neighbors (k=3) Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

metrics.ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test, cmap='Greens')
plt.title("KNN (k=3) - Confusion Matrix")
plt.show()

"""
Summary & Observations:

Top features most correlated with 'y':
- poutcome_success       (corr ≈ 0.28)
- poutcome_unknown       (corr ≈ 0.16)
- housing_yes            (corr ≈ 0.10)
- job_retired            (corr ≈ 0.08)
- marital_married        (corr ≈ 0.06)

The correlation values with the target variable are relatively
low, which suggests that no single variable strongly predicts the outcome.
Instead, the model likely benefits from combinations of features.

Model Comparison:

Logistic Regression:
- Accuracy: ~89.8%
- Strong performance on predicting "no" class (majority class)
- Low recall for "yes" class, meaning it misses a lot of actual positive cases

K-Nearest Neighbors (k=3):
- Accuracy: ~87.1%
- Slightly more false positives
- Similar issue with low recall for the minority class

Conclusion:
Logistic regression performs better overall, especially in terms of
accuracy and precision. However, both models struggle with predicting the
minority class ("yes"). This suggests that addressing class imbalance
(e.g. SMOTE, class weights) could help improve recall for the positive class.
"""
