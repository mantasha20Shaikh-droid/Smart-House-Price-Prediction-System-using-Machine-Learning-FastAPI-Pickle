# Smart House Price Prediction System
# Machine Learning Model Training

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import pickle

print("=" * 70)
print("        SMART HOUSE PRICE PREDICTION MODEL")
print("=" * 70)

# Load Dataset

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("\nDataset Loaded Successfully")
print("-" * 70)

print("Total Records :", len(X))
print("Total Features:", len(X.columns))

print("\nDataset Features")

for feature in X.columns:
    print("•", feature)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Random Forest Model

print("\nTraining Machine Learning Model...")

model = RandomForestRegressor(

    n_estimators=300,

    max_depth=20,

    random_state=42,

    n_jobs=-1

)

model.fit(X_train, y_train)

print("Training Completed Successfully!")

# Prediction

predictions = model.predict(X_test)

# Evaluation

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

train_accuracy = model.score(X_train, y_train)

test_accuracy = model.score(X_test, y_test)

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")
print(f"MAE               : {mae:.4f}")
print(f"MSE               : {mse:.4f}")
print(f"RMSE              : {rmse:.4f}")
print(f"R² Score          : {r2:.4f}")

# Feature Importance

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(importance.to_string(index=False))

# Save Model using Pickle

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n")
print("=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("Saved File : house_price_model.pkl")

print("\nProject Ready for FastAPI Deployment!")
