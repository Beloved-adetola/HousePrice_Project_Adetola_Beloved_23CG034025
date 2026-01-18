import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("train.csv")

# 2. Select allowed features
selected_features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood"
]

df = df[selected_features + ["SalePrice"]]

# 3. Handle missing values
df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].median(), inplace=True)
df["GarageCars"].fillna(df["GarageCars"].median(), inplace=True)

# 4. Encode categorical variable (Neighborhood)
le = LabelEncoder()
df["Neighborhood"] = le.fit_transform(df["Neighborhood"])

# 5. Split features and target
X = df[selected_features]
y = df["SalePrice"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Model training
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 9. Evaluation
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 10. Save model, scaler, encoder
with open("house_price_model.pkl", "wb") as f:
    pickle.dump((model, scaler, le), f)

print("Model saved successfully!")

# 11. Reload model (proof of persistence)
with open("house_price_model.pkl", "rb") as f:
    loaded_model, loaded_scaler, loaded_le = pickle.load(f)

sample = X_test.iloc[[0]]
sample_scaled = loaded_scaler.transform(sample)
print("Reloaded model prediction:", loaded_model.predict(sample_scaled))
