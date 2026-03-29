import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Load data
data_path = Path(__file__).resolve().parent / "student_data.xlsx"
if not data_path.exists():
    raise FileNotFoundError(f"Missing Excel file: {data_path}")

data = pd.read_excel(data_path)

# Features and targets
X = data[['Hours_Studied','Attendance','Previous_Marks','Sleep_Hours','Internet_Usage']]
y_reg = data['Final_Score']
y_clf = data['Pass']

# Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train_reg)
pred_lr = lr.predict(X_test)
print("MSE:", mean_squared_error(y_test_reg, pred_lr))

# Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train_clf)
pred_log = log.predict(X_test)
print("Logistic Accuracy:", accuracy_score(y_test_clf, pred_log))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train_clf)
pred_dt = dt.predict(X_test)
print("DT Accuracy:", accuracy_score(y_test_clf, pred_dt))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train_clf)
pred_rf = rf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test_clf, pred_rf))
