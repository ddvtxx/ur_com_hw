import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def part1_eda():
    """Part 1: Exploratory Data Analysis"""
    print("PART 1: EXPLORATORY DATA ANALYSIS")

    # Load data
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')

    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 1. Traffic volume over time
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['traffic_volume'], linewidth=0.5, alpha=0.8)
    plt.title('Traffic Volume Over Time')
    plt.ylabel('Traffic Volume')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Distribution of traffic volume
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df['traffic_volume'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Traffic Volume')
    plt.xlabel('Traffic Volume')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    df['traffic_volume'].plot(kind='box', vert=False)
    plt.title('Box Plot of Traffic Volume')
    plt.xlabel('Traffic Volume')
    plt.tight_layout()
    plt.show()

    # 3. Traffic by hour of day
    df['hour'] = df.index.hour
    hourly_traffic = df.groupby('hour')['traffic_volume'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_traffic.index, hourly_traffic.values, marker='o')
    plt.title('Average Traffic Volume by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Traffic Volume')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

    # 4. Traffic by day of week
    df['dayofweek'] = df.index.dayofweek
    daily_traffic = df.groupby('dayofweek')['traffic_volume'].mean()

    plt.figure(figsize=(10, 5))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.plot(days, daily_traffic.values, marker='o')
    plt.title('Average Traffic Volume by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Traffic Volume')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Weather analysis
    weather_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, feature in enumerate(weather_features):
        if feature == 'weather_main':
            df.boxplot(column='traffic_volume', by='weather_main', ax=axes[i])
            axes[i].set_title('Traffic Volume by Weather Main')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            axes[i].scatter(df[feature], df['traffic_volume'], alpha=0.3, s=1)
            axes[i].set_title(f'Traffic vs {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Traffic Volume')

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.suptitle('Traffic Volume vs Weather Features', y=1.02)
    plt.show()

    # 6. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Run Part 1
part1_eda()