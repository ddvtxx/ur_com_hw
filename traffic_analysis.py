import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 1. Data Loading and Preprocessing
def load_and_prep_data(filepath):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file is in the same directory.")
        return None

    # Handle missing AADT values by filling with 0 or dropping
    # We will use 'AHEAD_AADT' as the primary target, filling missing with 'BACK_AADT'
    df['Traffic_Volume'] = df['AHEAD_AADT'].fillna(df['BACK_AADT'])

    # Drop rows where target is still null
    df = df.dropna(subset=['Traffic_Volume'])

    # Select features for modeling
    # Using Coordinates (X, Y) and categorical features (ROUTE, COUNTY)
    features = ['X', 'Y', 'ROUTE', 'COUNTY']
    target = 'Traffic_Volume'

    data = df[features + [target]].copy()

    # Encode categorical variables
    le_county = LabelEncoder()
    data['COUNTY_ENC'] = le_county.fit_transform(data['COUNTY'].astype(str))

    return data, df  # Return processed data for ML and original for plotting


# 2. Spatial Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("Generating EDA plots...")

    # Plot 1: Spatial Distribution of Traffic Volume (Hotspots)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['X'], df['Y'],
                          c=df['Traffic_Volume'],
                          cmap='viridis',
                          alpha=0.5,
                          s=10)
    plt.colorbar(scatter, label='Annual Average Daily Traffic (AADT)')
    plt.title('Spatial Distribution of Traffic Volume in California')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.savefig('california_traffic_heatmap.png')
    plt.close()
    print(" - Saved 'california_traffic_heatmap.png'")

    # Plot 2: Distribution of Traffic Volume
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Traffic_Volume'], bins=50, kde=True, color='blue')
    plt.title('Distribution of AADT Across All Segments')
    plt.xlabel('Traffic Volume (AADT)')
    plt.ylabel('Frequency')
    plt.savefig('traffic_volume_distribution.png')
    plt.close()
    print(" - Saved 'traffic_volume_distribution.png'")

    # Plot 3: Top 10 Counties by Average Traffic
    county_avg = df.groupby('COUNTY')['Traffic_Volume'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=county_avg.index, y=county_avg.values, palette='magma')
    plt.title('Top 10 Counties by Average Highway Traffic Volume')
    plt.xlabel('County')
    plt.ylabel('Average AADT')
    plt.savefig('top_counties_traffic.png')
    plt.close()
    print(" - Saved 'top_counties_traffic.png'")


# 3. Predictive Modeling
def train_model(data):
    print("Training Predictive Model...")

    # Features and Target
    X = data[['X', 'Y', 'ROUTE', 'COUNTY_ENC']]
    y = data['Traffic_Volume']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Regressor
    # Random Forest works well for spatial data as it can capture non-linear geographic patterns
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R^2 Score: {r2:.4f}")

    # Plot 4: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='green')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual AADT')
    plt.ylabel('Predicted AADT')
    plt.title('Random Forest: Actual vs Predicted Traffic Volume')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    print(" - Saved 'actual_vs_predicted.png'")

    return rf_model


if __name__ == "__main__":
    # Filename based on user upload
    filename = 'Annual_Average_Daily_Traffic.csv'

    processed_data, original_df = load_and_prep_data(filename)

    if processed_data is not None:
        perform_eda(original_df)
        train_model(processed_data)
        print("\nAnalysis Complete. Report images saved.")