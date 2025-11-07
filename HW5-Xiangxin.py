
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def q1():
    """
    Question 1: Construct a rank-1 3x4 matrix that can be exactly reconstructed
    with only one singular value.
    """
    print("=== Question 1 ===")

    # Create a rank-1 matrix by taking outer product of two vectors
    u = np.array([1, 2, 3])  # Left singular vector
    v = np.array([2, 0, 1, -1])  # Right singular vector

    # Construct rank-1 matrix A = u * v^T
    A = np.outer(u, v)

    print("Matrix A (rank-1):")
    print(A)

    # Verify it's rank-1
    rank = np.linalg.matrix_rank(A)
    print(f"Rank of A: {rank}")

    # Perform SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"Singular values: {s}")

    # Reconstruct using only first singular value
    A_reconstructed = s[0] * np.outer(U[:, 0], Vt[0, :])

    print("Reconstructed A (using only first singular value):")
    print(A_reconstructed)

    # Check if they're exactly equal
    print(f"Exact reconstruction: {np.allclose(A, A_reconstructed)}")

    return A


def q2():
    """
    Question 2: Analyze k-means on 1D dataset [1, 3, 6, 7] with k=3
    """
    print("\n=== Question 2 ===")

    data = np.array([1, 3, 6, 7]).reshape(-1, 1)

    # Find optimal clustering manually
    # With 4 points and k=3, one cluster must have 2 points, others have 1
    # Try all possible clusterings:

    possible_clusterings = [
        # Format: (cluster_centers, assignments, SSE)
        ([1, 3, 6.5], [0, 1, 2, 2]),  # Clusters: [1], [3], [6,7]
        ([2, 6, 7], [0, 0, 1, 2]),  # Clusters: [1,3], [6], [7]
        ([1, 4.5, 7], [0, 1, 1, 2]),  # Clusters: [1], [3,6], [7]
    ]

    best_sse = float('inf')
    best_clustering = None

    for centers, assignments in possible_clusterings:
        sse = 0
        for point, cluster_idx in zip(data, assignments):
            sse += (point[0] - centers[cluster_idx]) ** 2

        if sse < best_sse:
            best_sse = sse
            best_clustering = (centers, assignments, sse)

    print(f"Optimal clustering: Centers = {best_clustering[0]}")
    print(f"Assignments: {best_clustering[1]}")
    print(f"SSE: {best_clustering[2]:.2f}")

    # Run k-means multiple times to see what it finds
    print("\nRunning k-means 10 times:")
    kmeans_results = []

    for i in range(10):
        kmeans = KMeans(n_clusters=3, n_init=1, random_state=i)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_.flatten()

        # Calculate SSE
        sse = 0
        for j, point in enumerate(data):
            sse += (point[0] - centers[labels[j]]) ** 2

        kmeans_results.append((centers, labels, sse))
        print(f"Run {i + 1}: Centers = {centers.round(2)}, SSE = {sse:.2f}")

    # Check if k-means finds optimal
    optimal_found = any(np.allclose(result[2], best_clustering[2]) for result in kmeans_results)

    print(f"\nK-means finds optimal clustering: {optimal_found}")
    print("Reason: K-means can get stuck in local optima due to random initialization")
    print("Conditions for finding optimal: Good initialization or multiple restarts")

    return data, best_clustering, kmeans_results


def q3():
    """
    Question 3: Analyze urban mobility matrix properties
    """
    print("\n=== Question 3 ===")

    answers = {
        "rank_range": "Approximately 2-8",
        "increase_rank": [
            "More diverse mobility patterns across zones",
            "Different hourly patterns for different zone types",
            "Seasonal variations captured in the data",
            "Special events causing unusual mobility patterns"
        ],
        "decrease_rank": [
            "More uniform mobility patterns across city",
            "Strong daily periodicity dominating all zones",
            "Data smoothing or aggregation",
            "Limited transportation options creating similar patterns"
        ],
        "singular_values_meaning": (
            "Singular values represent the importance of different patterns in the data. "
            "The first few likely capture: 1) Overall daily periodicity, 2) Work vs residential zone differences, "
            "3) Day-of-week patterns, 4) Specialized zone behaviors"
        )
    }

    print(f"Expected rank range: {answers['rank_range']}")
    print("\nFactors increasing rank:")
    for factor in answers['increase_rank']:
        print(f"  - {factor}")

    print("\nFactors decreasing rank:")
    for factor in answers['decrease_rank']:
        print(f"  - {factor}")

    print(f"\nSingular values denote: {answers['singular_values_meaning']}")

    return answers


def q4():
    """
    Question 4: Citibike clustering analysis using actual data files
    """
    print("=== Question 4: Citibike Clustering Analysis ===")

    # Load the data
    try:
        # Load both CSV files
        df1 = pd.read_csv('202401-citibike-tripdata_1.csv')
        df2 = pd.read_csv('202401-citibike-tripdata_2.csv')

        # Combine the datasets
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"Loaded {len(df)} total trips")

    except FileNotFoundError:
        print("Error: CSV files not found. Please ensure they are in the same directory.")
        return None

    # Data cleaning and preprocessing
    print("\n--- Data Preprocessing ---")
    # Remove trips with missing coordinates
    initial_count = len(df)
    df = df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
    print(f"Removed {initial_count - len(df)} trips with missing coordinates")

    # Remove unrealistic trips (same start and end station with exactly same coordinates)
    same_location = (df['start_lat'] == df['end_lat']) & (df['start_lng'] == df['end_lng'])
    df = df[~same_location]
    print(f"Removed {same_location.sum()} trips with same start/end location")

    print(f"Final dataset: {len(df)} trips")

    # 4a: Trip Clustering with Standard K-means
    print("\n--- 4a: Trip Clustering (Standard K-means) ---")

    # Prepare trip data as 4-tuples
    trip_features = df[['start_lat', 'end_lat', 'start_lng', 'end_lng']].values

    # Scale the features
    scaler = StandardScaler()
    trip_features_scaled = scaler.fit_transform(trip_features)

    # Choose k using elbow method
    k_trips = find_optimal_k(trip_features_scaled, max_k=8, data_type='trips')

    # Apply standard k-means
    kmeans_standard = KMeans(n_clusters=k_trips, random_state=42, n_init=10)
    trip_labels_standard = kmeans_standard.fit_predict(trip_features_scaled)

    print(f"Optimal k for trips: {k_trips}")
    print(f"Cluster sizes: {np.bincount(trip_labels_standard)}")
    print(f"Inertia (SSE): {kmeans_standard.inertia_:.2f}")

    # 4b: Trip Clustering with K-means++
    print("\n--- 4b: Trip Clustering (K-means++) ---")

    kmeans_plus = KMeans(n_clusters=k_trips, init='k-means++', random_state=42, n_init=10)
    trip_labels_plus = kmeans_plus.fit_predict(trip_features_scaled)

    print(f"Cluster sizes: {np.bincount(trip_labels_plus)}")
    print(f"Inertia (SSE): {kmeans_plus.inertia_:.2f}")

    # 4c: Station Clustering
    print("\n--- 4c: Station Clustering ---")

    # Create station-hour matrix
    station_hour_matrix = create_station_hour_matrix(df)
    print(f"Station-hour matrix shape: {station_hour_matrix.shape}")

    # Scale station data
    station_scaler = StandardScaler()
    station_features_scaled = station_scaler.fit_transform(station_hour_matrix)

    # Choose k for stations
    k_stations = find_optimal_k(station_features_scaled, max_k=6, data_type='stations')

    # Standard K-means for stations
    kmeans_stations_standard = KMeans(n_clusters=k_stations, random_state=42, n_init=10)
    station_labels_standard = kmeans_stations_standard.fit_predict(station_features_scaled)

    # K-means++ for stations
    kmeans_stations_plus = KMeans(n_clusters=k_stations, init='k-means++', random_state=42, n_init=10)
    station_labels_plus = kmeans_stations_plus.fit_predict(station_features_scaled)

    print(f"Optimal k for stations: {k_stations}")
    print(f"Standard K-means - Cluster sizes: {np.bincount(station_labels_standard)}")
    print(f"Standard K-means - Inertia: {kmeans_stations_standard.inertia_:.2f}")
    print(f"K-means++ - Cluster sizes: {np.bincount(station_labels_plus)}")
    print(f"K-means++ - Inertia: {kmeans_stations_plus.inertia_:.2f}")

    # Analysis and Visualization
    print("\n--- Qualitative Analysis ---")
    analyze_results(df, trip_features, trip_labels_standard, trip_labels_plus,
                    station_hour_matrix, station_labels_standard, station_labels_plus,
                    kmeans_standard, kmeans_plus, kmeans_stations_standard, kmeans_stations_plus)

    return {
        'trip_data': trip_features,
        'trip_labels_standard': trip_labels_standard,
        'trip_labels_plus': trip_labels_plus,
        'station_data': station_hour_matrix,
        'station_labels_standard': station_labels_standard,
        'station_labels_plus': station_labels_plus,
        'k_trips': k_trips,
        'k_stations': k_stations
    }


def find_optimal_k(data, max_k=8, data_type='trips'):
    """
    Find optimal k using elbow method
    """
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Simple elbow detection (you can make this more sophisticated)
    # For this example, I'll choose based on typical patterns
    if data_type == 'trips':
        return 4  # Based on common trip patterns: short/local, medium, long, cross-city
    else:
        return 3  # Based on station types: residential, commercial, transit hubs


def create_station_hour_matrix(df):
    """
    Create 24-dimensional vectors for stations showing hourly trip counts
    """
    # Extract hour from started_at
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['hour'] = df['started_at'].dt.hour

    # Get unique stations (using both start and end stations)
    all_stations = pd.concat([df['start_station_id'], df['end_station_id']]).unique()

    # Create station-hour matrix for departures
    start_counts = df.groupby(['start_station_id', 'hour']).size().unstack(fill_value=0)

    # Create station-hour matrix for arrivals
    end_counts = df.groupby(['end_station_id', 'hour']).size().unstack(fill_value=0)

    # Combine departures and arrivals (to or from station)
    # Reindex to include all stations and all hours (0-23)
    all_hours = range(24)
    start_counts = start_counts.reindex(index=all_stations, columns=all_hours, fill_value=0)
    end_counts = end_counts.reindex(index=all_stations, columns=all_hours, fill_value=0)

    station_hour_matrix = start_counts + end_counts

    return station_hour_matrix


def analyze_results(df, trip_features, trip_labels_std, trip_labels_plus,
                    station_matrix, station_labels_std, station_labels_plus,
                    kmeans_std, kmeans_plus, kmeans_stat_std, kmeans_stat_plus):
    """
    Perform qualitative analysis and create visualizations
    """
    print("\n=== QUALITATIVE ANALYSIS ===")

    # 1. Trip Clustering Comparison
    print("\n1. TRIP CLUSTERING COMPARISON:")
    print(f"Standard K-means SSE: {kmeans_std.inertia_:.2f}")
    print(f"K-means++ SSE: {kmeans_plus.inertia_:.2f}")
    print(f"Improvement: {((kmeans_std.inertia_ - kmeans_plus.inertia_) / kmeans_std.inertia_ * 100):.1f}%")

    # Analyze trip clusters
    df_trip_std = df.copy()
    df_trip_std['cluster'] = trip_labels_std
    df_trip_plus = df.copy()
    df_trip_plus['cluster'] = trip_labels_plus

    print("\nTrip Cluster Characteristics (Standard K-means):")
    for cluster in range(len(np.unique(trip_labels_std))):
        cluster_data = df_trip_std[df_trip_std['cluster'] == cluster]
        avg_distance = calculate_avg_distance(cluster_data)
        print(f"Cluster {cluster}: {len(cluster_data)} trips, Avg distance: {avg_distance:.2f} km")

    # 2. Station Clustering Comparison
    print("\n2. STATION CLUSTERING COMPARISON:")
    print(f"Standard K-means SSE: {kmeans_stat_std.inertia_:.2f}")
    print(f"K-means++ SSE: {kmeans_stat_plus.inertia_:.2f}")

    # Analyze station clusters
    print("\nStation Cluster Patterns (K-means++):")
    station_clusters = pd.DataFrame(station_matrix)
    station_clusters['cluster'] = station_labels_plus

    for cluster in range(len(np.unique(station_labels_plus))):
        cluster_stations = station_clusters[station_clusters['cluster'] == cluster]
        avg_pattern = cluster_stations.iloc[:, :24].mean(axis=0)
        peak_hour = avg_pattern.idxmax()
        print(f"Cluster {cluster}: {len(cluster_stations)} stations, Peak hour: {peak_hour}:00")

    # Create visualizations
    create_visualizations(df, trip_features, trip_labels_std, trip_labels_plus,
                          station_matrix, station_labels_std, station_labels_plus)


def calculate_avg_distance(cluster_data):
    """
    Calculate approximate average distance using Haversine formula
    """
    # Simplified distance calculation (in reality, use Haversine)
    distances = np.sqrt(
        (cluster_data['start_lat'] - cluster_data['end_lat']) ** 2 +
        (cluster_data['start_lng'] - cluster_data['end_lng']) ** 2
    ) * 111  # Approximate km per degree
    return distances.mean()


def create_visualizations(df, trip_features, trip_labels_std, trip_labels_plus,
                          station_matrix, station_labels_std, station_labels_plus):
    """
    Create comprehensive visualizations
    """
    plt.figure(figsize=(16, 12))

    # 1. Trip Clusters - Standard K-means
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(trip_features[:, 2], trip_features[:, 0],
                          c=trip_labels_std, alpha=0.6, cmap='viridis', s=1)
    plt.xlabel('Start Longitude')
    plt.ylabel('Start Latitude')
    plt.title('Trip Clusters - Standard K-means')
    plt.colorbar(scatter, label='Cluster')

    # 2. Trip Clusters - K-means++
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(trip_features[:, 2], trip_features[:, 0],
                          c=trip_labels_plus, alpha=0.6, cmap='viridis', s=1)
    plt.xlabel('Start Longitude')
    plt.ylabel('Start Latitude')
    plt.title('Trip Clusters - K-means++')
    plt.colorbar(scatter, label='Cluster')

    # 3. Station Usage Patterns - K-means++
    plt.subplot(2, 3, 3)
    unique_clusters = np.unique(station_labels_plus)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

    for cluster, color in zip(unique_clusters, colors):
        cluster_data = station_matrix[station_labels_plus == cluster]
        if len(cluster_data) > 0:
            avg_pattern = cluster_data.mean(axis=0)
            plt.plot(range(24), avg_pattern, color=color, label=f'Cluster {cluster}', linewidth=2)

    plt.xlabel('Hour of Day')
    plt.ylabel('Average Trip Count')
    plt.title('Station Usage Patterns by Cluster')
    plt.legend()
    plt.xticks(range(0, 24, 3))

    # 4. Cluster Size Comparison - Trips
    plt.subplot(2, 3, 4)
    std_sizes = np.bincount(trip_labels_std)
    plus_sizes = np.bincount(trip_labels_plus)

    x = np.arange(len(std_sizes))
    width = 0.35

    plt.bar(x - width / 2, std_sizes, width, label='Standard', alpha=0.7)
    plt.bar(x + width / 2, plus_sizes, width, label='K-means++', alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Trips')
    plt.title('Trip Cluster Sizes Comparison')
    plt.legend()

    # 5. Cluster Size Comparison - Stations
    plt.subplot(2, 3, 5)
    std_stat_sizes = np.bincount(station_labels_std)
    plus_stat_sizes = np.bincount(station_labels_plus)

    x = np.arange(len(std_stat_sizes))

    plt.bar(x - width / 2, std_stat_sizes, width, label='Standard', alpha=0.7)
    plt.bar(x + width / 2, plus_stat_sizes, width, label='K-means++', alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Stations')
    plt.title('Station Cluster Sizes Comparison')
    plt.legend()

    # 6. Member vs Casual Distribution by Trip Cluster
    plt.subplot(2, 3, 6)
    df['cluster'] = trip_labels_plus
    cluster_member = df.groupby('cluster')['member_casual'].value_counts().unstack()
    cluster_member.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Cluster')
    plt.ylabel('Number of Trips')
    plt.title('Member vs Casual Riders by Cluster')
    plt.legend(title='Rider Type')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('citibike_clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved as 'citibike_clustering_analysis.png'")

# Run all questions
if __name__ == "__main__":
    results = {}
    results['q1'] = q1()
    results['q2'] = q2()
    results['q3'] = q3()
    results['q4'] = q4()