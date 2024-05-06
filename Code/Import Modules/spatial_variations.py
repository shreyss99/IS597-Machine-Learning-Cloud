import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import geopandas
import geodatasets


def find_optimal_clusters(dataframe, pipeline, max_k):
    """
    Find the optimal number of clusters using the elbow method.

    Parameters:
    dataframe: The DataFrame containing the data.
    pipeline: The pipeline object containing the clustering algorithm.
    max_k: The maximum number of clusters to consider.

    Returns:
    int: The optimal number of clusters.
    """

    wcss = []
    for k in range(1, max_k + 1):
        kmeans_pipeline = pipeline
        kmeans_pipeline.steps[-1] = ('kmeans', KMeans(n_clusters=k, init='k-means++'))
        kmeans_pipeline.fit(dataframe)
        wcss.append(kmeans_pipeline['kmeans'].inertia_)

    # Plotting the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Plot')

    # Find the knee point
    kl = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
    optimal_clusters = kl.elbow
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label='Optimal Clusters')
    plt.legend()
    plt.show()

    return kl.elbow


def create_clusters(dataframe, latitude_feature, longitude_feature, address_feature):
    """
    Create clusters of data points based on latitude and longitude.

    Parameters:
    dataframe: The DataFrame containing the data.
    latitude_feature: The name of the column containing latitude values.
    longitude_feature: The name of the column containing longitude values.
    address_feature: The name of the column containing address information.

    Returns:
    pandas.DataFrame: The DataFrame with an additional column indicating cluster labels.
    """

    print("************** TRAINING CLUSTER MODEL **************", "\n")

    # Filter out rows with non-zero coordinates
    non_zero_coordinates_data = dataframe[(dataframe[latitude_feature] != 0) & (dataframe[longitude_feature] != 0)]

    # Select relevant columns for clustering
    clustering_data = non_zero_coordinates_data[[latitude_feature, longitude_feature, address_feature]]

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [latitude_feature, longitude_feature]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [address_feature])
        ],
        remainder='passthrough'
    )

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply preprocessing
    clustering_data_processed = pipeline.fit_transform(clustering_data)

    # Initialize KMeans++
    kmeans_pp = KMeans(init='k-means++')

    # Pipeline for KMeans
    pipeline_kmeans = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', kmeans_pp)])

    print("************** IDENTIFYING THE OPTIMAL NUMBER OF CLUSTERS **************", "\n")

    # Choose the optimal number of clusters (you can use the elbow method plot to determine this)
    optimal_k = find_optimal_clusters(clustering_data, pipeline_kmeans, max_k=10)

    print("Optimal number of clusters:", optimal_k)

    # Fit KMeans with optimal number of clusters
    kmeans_pp = KMeans(n_clusters=optimal_k, init='k-means++')
    kmeans_pp.fit(clustering_data_processed)

    # Assign cluster labels to the original DataFrame
    non_zero_coordinates_data.loc[:, 'CLUSTER'] = kmeans_pp.labels_

    print("************** PLOTTING THE CLUSTERS **************", "\n")

    # Load Chicago map data
    chicago_map = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))

    # Plot Chicago map
    fig, ax = plt.subplots(figsize=(10, 8))
    chicago_map.plot(ax=ax, color='lightgrey', edgecolor='black')

    # Plot clusters
    for cluster_label in range(optimal_k):
        cluster_data = non_zero_coordinates_data[non_zero_coordinates_data['CLUSTER'] == cluster_label]
        ax.scatter(cluster_data[longitude_feature], cluster_data[latitude_feature], label=f'Cluster {cluster_label}',
                   alpha=0.5)

    # Set plot title and labels
    ax.set_title('Clusters of Speed Camera Violations')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.grid(True)

    # Adjust x-axis and y-axis limits for zooming
    ax.set_xlim([-87.90, -87.5])
    ax.set_ylim([41.6, 42.05])

    plt.show()
