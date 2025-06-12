import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


df = pd.read_csv('Training_Strategy_Document.txt', sep='\t', engine='python')

df = df.dropna(subset=['Mag'])

features = ['Latitude', 'Longitude', 'Depth']
X = df[features]
y = df['Mag']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mean Predictor
mean_pred = np.full_like(y_test, y_train.mean(), dtype=np.float64)
mae_mean = mean_absolute_error(y_test, mean_pred)
rmse_mean = np.sqrt(mean_squared_error(y_test, mean_pred))

# Median Predictor
median_pred = np.full_like(y_test, y_train.median(), dtype=np.float64)
mae_median = mean_absolute_error(y_test, median_pred)
rmse_median = np.sqrt(mean_squared_error(y_test, median_pred))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
mae_lr = mean_absolute_error(y_test, lr_pred)
rmse_lr = np.sqrt(mean_squared_error(y_test, lr_pred))
r2_lr = r2_score(y_test, lr_pred)

# results for report
print("=== Baseline Model Results ===")
print(f"Mean Predictor:    MAE={mae_mean:.3f}, RMSE={rmse_mean:.3f}")
print(f"Median Predictor:  MAE={mae_median:.3f}, RMSE={rmse_median:.3f}")
print(f"Linear Regression: MAE={mae_lr:.3f}, RMSE={rmse_lr:.3f}, R2={r2_lr:.3f}")
