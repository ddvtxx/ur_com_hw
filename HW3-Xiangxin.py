import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load Boston Housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Basic exploration
print("\nDataset info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Distribution of target variable MEDV
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.hist(df['MEDV'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of MEDV (Target)')
plt.xlabel('MEDV')
plt.ylabel('Frequency')

# Correlation analysis
correlation_matrix = df.corr()
plt.subplot(3, 3, 2)
sns.heatmap(correlation_matrix[['MEDV']].sort_values('MEDV', ascending=False),
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation with MEDV')
plt.tight_layout()

# Top correlated features with MEDV
corr_with_medv = correlation_matrix['MEDV'].sort_values(ascending=False)
print("\nTop correlated features with MEDV:")
print(corr_with_medv)

# Select top features based on correlation (excluding MEDV itself)
top_features = corr_with_medv.index[1:6]  # Top 5 features excluding MEDV
print(f"\nSelected features: {list(top_features)}")

# Feature distributions vs MEDV
for i, feature in enumerate(top_features[:4], 3):
    plt.subplot(3, 3, i)
    plt.scatter(df[feature], df['MEDV'], alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel('MEDV')
    plt.title(f'{feature} vs MEDV')

plt.tight_layout()
plt.show()

# Prepare data for modeling
X = df[top_features]
y = df['MEDV']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Part a: Linear Regression with selected features
print("\n" + "="*50)
print("PART A: LINEAR REGRESSION")
print("="*50)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Results:")
print(f"MSE: {mse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print(f"Coefficients: {dict(zip(top_features, lr.coef_))}")
print(f"Intercept: {lr.intercept_:.4f}")

# Part b: Ridge Regression
print("\n" + "="*50)
print("PART B: RIDGE REGRESSION")
print("="*50)

# Find optimal alpha for Ridge
alphas = np.logspace(-3, 3, 100)
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_ridge)
    ridge_scores.append(r2)

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
print(f"Optimal alpha for Ridge: {best_alpha_ridge:.4f}")

# Train with optimal alpha
ridge = Ridge(alpha=best_alpha_ridge, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression Results:")
print(f"MSE: {mse_ridge:.4f}")
print(f"R² Score: {r2_ridge:.4f}")
print(f"Coefficients: {dict(zip(top_features, ridge.coef_))}")
print(f"Intercept: {ridge.intercept_:.4f}")

# Part c: LASSO Regression
print("\n" + "="*50)
print("PART C: LASSO REGRESSION")
print("="*50)

# Find optimal alpha for LASSO
lasso_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_lasso)
    lasso_scores.append(r2)

best_alpha_lasso = alphas[np.argmax(lasso_scores)]
print(f"Optimal alpha for LASSO: {best_alpha_lasso:.4f}")

# Train with optimal alpha
lasso = Lasso(alpha=best_alpha_lasso, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"LASSO Regression Results:")
print(f"MSE: {mse_lasso:.4f}")
print(f"R² Score: {r2_lasso:.4f}")
print(f"Coefficients: {dict(zip(top_features, lasso.coef_))}")
print(f"Intercept: {lasso.intercept_:.4f}")

# Comparison and Visualization
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results = {
    'Model': ['Linear Regression', 'Ridge Regression', 'LASSO Regression'],
    'MSE': [mse_lr, mse_ridge, mse_lasso],
    'R² Score': [r2_lr, r2_ridge, r2_lasso],
    'Optimal Alpha': ['N/A', f'{best_alpha_ridge:.4f}', f'{best_alpha_lasso:.4f}']
}

results_df = pd.DataFrame(results)
print(results_df)

# Coefficient comparison
coefficients_df = pd.DataFrame({
    'Feature': top_features,
    'Linear Regression': lr.coef_,
    'Ridge Regression': ridge.coef_,
    'LASSO Regression': lasso.coef_
})

print("\nCoefficient Comparison:")
print(coefficients_df)

# Visualization
plt.figure(figsize=(18, 12))

# 1. Actual vs Predicted plots
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='blue', label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()

plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.7, color='red', label='Ridge Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Ridge Regression: Actual vs Predicted')
plt.legend()

plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_lasso, alpha=0.7, color='green', label='LASSO Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('LASSO Regression: Actual vs Predicted')
plt.legend()

# 2. R² scores comparison
plt.subplot(2, 3, 4)
models = ['Linear', 'Ridge', 'LASSO']
r2_scores = [r2_lr, r2_ridge, r2_lasso]
colors = ['blue', 'red', 'green']
plt.bar(models, r2_scores, color=colors, alpha=0.7)
plt.ylabel('R² Score')
plt.title('Model Performance Comparison (R²)')
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 3. MSE comparison
plt.subplot(2, 3, 5)
mse_scores = [mse_lr, mse_ridge, mse_lasso]
plt.bar(models, mse_scores, color=colors, alpha=0.7)
plt.ylabel('MSE')
plt.title('Model Performance Comparison (MSE)')
for i, v in enumerate(mse_scores):
    plt.text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom')

# 4. Coefficient magnitudes
plt.subplot(2, 3, 6)
x_pos = np.arange(len(top_features))
width = 0.25

plt.bar(x_pos - width, lr.coef_, width, label='Linear', alpha=0.7)
plt.bar(x_pos, ridge.coef_, width, label='Ridge', alpha=0.7)
plt.bar(x_pos + width, lasso.coef_, width, label='LASSO', alpha=0.7)

plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Comparison')
plt.xticks(x_pos, top_features, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Residual analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
residuals_lr = y_test - y_pred_lr
plt.scatter(y_pred_lr, residuals_lr, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Linear Regression Residuals')

plt.subplot(1, 3, 2)
residuals_ridge = y_test - y_pred_ridge
plt.scatter(y_pred_ridge, residuals_ridge, alpha=0.7, color='red')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Ridge Regression Residuals')

plt.subplot(1, 3, 3)
residuals_lasso = y_test - y_pred_lasso
plt.scatter(y_pred_lasso, residuals_lasso, alpha=0.7, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('LASSO Regression Residuals')

plt.tight_layout()
plt.show()
