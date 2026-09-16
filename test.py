import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from scipy.stats import zscore

import joblib
import warnings

warnings.filterwarnings("ignore")

print("Libraries imported successfully")


# --------------------------------
# 1. Load Iris Dataset
# --------------------------------

Cancer=load_breast_cancer()

df = pd.DataFrame(
    Cancer.data,
    columns=Cancer.feature_names
)

# Add target column
df["target"] = Cancer.target

print("Dataset:")
print(df.head())

print()


# --------------------------------
# 2. Remove Outliers
# --------------------------------

z = np.abs(zscore(df))

dfn = df[(z < 3).all(axis=1)]

print("Old shape:", df.shape)
print("New shape:", dfn.shape)

print()


# --------------------------------
# 3. Separate X and Y
# --------------------------------

# X = four input features
X = dfn.iloc[:, :-1]

# Y = target
y = dfn.iloc[:, -1]

print("X shape:", X.shape)
print("Y shape:", y.shape)

print()


# --------------------------------
# 4. Train Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print()


# --------------------------------
# 5. Create SVC Model
# --------------------------------

svc = SVC()

# Train model
svc.fit(X_train, y_train)

print("Model trained successfully")


# --------------------------------
# 6. Prediction
# --------------------------------

y_pred = svc.predict(X_test)


# --------------------------------
# 7. Accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# --------------------------------
# 8. Classification Report
# --------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=Cancer.target_names
    )
)


# --------------------------------
# 9. Confusion Matrix
# --------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------
# 10. Save Model
# --------------------------------

joblib.dump(svc, "svc.pkl")

print("\nModel saved successfully as svc.pkl")