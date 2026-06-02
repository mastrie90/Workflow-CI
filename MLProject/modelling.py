import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ✅ MLflow config
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Default")

# ✅ Autolog (WAJIB kalau mau simpel)
mlflow.sklearn.autolog()

# ✅ Load data
df = pd.read_csv("dataset_clean.csv")

# ✅ Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# ✅ Split fitur & target
X = df.drop("price", axis=1)
y = df["price"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ TRAINING + LOGGING
with mlflow.start_run(run_name="RandomForest_Run"):

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ✅ Predict
    y_pred = model.predict(X_test)

    # ✅ Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("✅ RMSE:", rmse)
    print("✅ R2:", r2)

    # ✅ (Optional manual logging tambahan)
    mlflow.log_metric("rmse_manual", rmse)
    mlflow.log_metric("r2_manual", r2)

print("✅ Training selesai & sudah masuk MLflow!")
