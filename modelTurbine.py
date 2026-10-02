import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
import joblib
import numpy as np

#Load dataset previously cleaned
df=pd.read_csv('data/dataTurbine_clean.csv')

#Select features to be included in the model
features = ['WindSpeed', 'WindDirection', 'Wind_N-S', 'Wind_E-W', 'Windp2', 'Windp3']
X = df[features]
y = df.PowerOutput
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=False)

#Parameters optimised using RandomizedSearchCV
params = {
    "n_estimators": 152,
    "max_depth": 7,
    "min_samples_split": 6,
    "learning_rate": 0.08,
    "subsample": 0.78,
    "min_samples_leaf": 8,
    "min_samples_split": 6,
    "loss": "squared_error",
}
reg = GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

#Printing metrics
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} kW")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")


#Save the model
joblib.dump(reg, 'modelTurbine.pkl')
