from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def part2_regression():
    """Part 2: Regression Models"""
    print("\nPART 2: REGRESSION MODELS")

    # Load and prepare data
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')

    # Create time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Define features and target
    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']

    # Chronological split (80% train, 20% test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Preprocessing
    numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    categorical_features = ['holiday', 'weather_main', 'is_weekend']
    cyclical_features = ['hour_sin', 'hour_cos']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('pass', 'passthrough', cyclical_features + ['dayofweek', 'month', 'year'])
        ])

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    # Time series cross-validation (k=5)
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        # Train and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'CV_RMSE_mean': cv_rmse.mean(),
            'CV_RMSE_std': cv_rmse.std(),
            'Test_RMSE': rmse,
            'Test_MAE': mae,
            'Test_R2': r2
        }

    # Display results
    results_df = pd.DataFrame(results).T
    print("\n=== REGRESSION RESULTS ===")
    print(results_df.round(4))

    # Feature importance
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)

    feature_names = (numeric_features +
                     list(rf_pipeline.named_steps['preprocessor']
                          .named_transformers_['cat']
                          .get_feature_names_out(categorical_features)) +
                     cyclical_features + ['dayofweek', 'month', 'year'])

    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.show()

    return results_df


# Run Part 2
regression_results = part2_regression()