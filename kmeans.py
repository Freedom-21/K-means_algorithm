import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# calculates the euclidean distance to  measure the distance between a data point and a centroid
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# The next two measurments  are for elbow method 

# Finding the distance between a set of points and a centroid
# To measure the compactness of a cluster 
def find_distance_to_centroid(points, centroid):
    distances = []
    for point in points:
        distances.append(euclidean_distance(point, centroid))
    return np.mean(distances)

#  calculate the average distance between all points in each cluster and their respective centroid
#  To evaluate the overall compactness of the clusters.
def find_average_distance(clusters, centroids):
    total_distance = 0
    for cluster_idx, cluster in enumerate(clusters):
        total_distance += find_distance_to_centroid(cluster, centroids[cluster_idx])
    return total_distance / len(centroids)

def plot_clusters_2D(clusters, centroids):
    fig, ax = plt.subplots()
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i + 1}")
    centroids = np.array(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label="Centroids")
    ax.set_xlabel("Feature1")
    ax.set_ylabel("Feature2")
    ax.legend()
    plt.show()
    
# It call k_means function to cluster the data for each value of k 
# and call  compute_silhouette_scores function to calculate the silhouette score for each clustering
def silhouette_scores(data, max_k=10, max_iterations=100):
    
    '''
    Calculates and plots the silhouette scores for different values of k (number of clusters).

    Parameters:
        data (ndarray): The dataset to cluster.
        max_k (int): The maximum value of k (number of clusters) to consider. Defaults to 10.
        max_iterations (int): The maximum number of iterations to run the k-means algorithm. Defaults to 100.

    Returns:
        None
    '''
    
    scores = []
    K_values = list(range(2, max_k + 1))
    for k in K_values:
        clusters, centroids = k_means(k, data, max_iterations)
        score = compute_silhouette_scores(data, clusters, centroids)
        scores.append(score)
    plt.plot(K_values, scores, 'bo-')
    plt.xlabel('K Value')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different K Values')
    avg_score = np.mean(scores)
    plt.axhline(avg_score, color='r', linestyle='dashed', label=f'Average Silhouette Score: {avg_score:.2f}')
    plt.legend()
    plt.show()

def elbow_method(data, max_k=10, max_iterations=100):
    
    '''
    Calculates and plots the average distance between all points in each cluster
    and their respective centroid for different values of k (number of clusters) 
    to help determine the optimal number of clusters using the elbow method.

    Parameters:
        data (ndarray): The dataset to cluster.
        max_k (int): The maximum value of k (number of clusters) to consider. Defaults to 10.
        max_iterations (int): The maximum number of iterations to run the k-means algorithm. Defaults to 100.

    Returns:
        None
    '''
    
    # store the average distance between all points in each cluster and their respective centroid for each value of k
    average_distances = []
    K_values = list(range(1, max_k + 1))
    for k in K_values:
        clusters, centroids = k_means(k, data, max_iterations) # call k_means function to cluster the data into k clusters
        # calculate average distance between all points in each cluster and their respective centroid.
        # using find_average_distance function and appends it to the average_distances list
        average_distances.append(find_average_distance(clusters, centroids))
    plt.plot(K_values, average_distances, 'bo-')
    plt.xlabel('K Value')
    plt.ylabel('Average Distance to Centroid')
    plt.title('Elbow Method for Optimal K Value')
    plt.show()
    
## The main k means logic    
def k_means(k, data, max_iterations=100):
    
    '''
    Performs the k-means clustering algorithm on the given data.

    Parameters:
        k (int): The number of clusters to create.
        data (ndarray): The dataset to cluster.
        max_iterations (int): The maximum number of iterations to run the algorithm. Defaults to 100.

    Returns:
        clusters (list): A list of clusters, where each cluster is a list of data points.
        centroids (ndarray): The final centroids of the clusters.
    '''
    
    # first step in the k-means algorithm -> randomly initialize k centroids
    n = len(data)
    centroids = data[np.random.randint(0, n, k)]
    # starts a loop that will run for a maximum of max_iterations iterations
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        # starting a loop that iterates over each data point d in the dataset
        for d in data:
            # calculate the euclidean distance between the current data point d and each centroid c
            distances = [euclidean_distance(d, c) for c in centroids] 
            # find index of the centroid with the smallest distance to the current data point d.
            closest_centroid_index = np.argmin(distances)
            #  assign the current data point d to the cluster corresponding to the closest centroid.
            clusters[closest_centroid_index].append(d)
        empty_clusters = [idx for idx, cluster in enumerate(clusters) if len(cluster) == 0] # checks if there are any empty clusters
        if empty_clusters: # If there are any empty clusters,
            for idx in empty_clusters:
                centroids[idx] = data[np.random.randint(0, n)]
            continue # reassigns their centroids to be random points from the data and continues to the next iteration of the loop.
        prev_centroids = centroids
        # updates each centroid to be the mean of all points assigned to its corresponding cluster
        centroids = [np.mean(c, axis=0) for c in clusters]
        # checks if the centroids have stopped changing
        if np.isclose(prev_centroids, centroids).all():
            break # if they do, breaks out of the loop
    return clusters, centroids


# calculate the silhouette score for each point in the dataset and returns the mean silhouette score
def compute_silhouette_scores(data, clusters, centroids): 
    
    '''
    Calculates the silhouette score for a given clustering of the data.

    Parameters:
        data (ndarray): The dataset that has been clustered.
        clusters (list): A list of clusters, where each cluster is a list of data points.
        centroids (ndarray): The centroids of the clusters.

    Returns:
        silhouette_score (float): The mean silhouette score for the given clustering.
    '''
    
    # initializing  empty array with the same shape as the data to store the cluster label for each point in the dataset.
    cluster_labels = np.empty_like(data) 
    # iterate over each cluster and each point in each cluster
    # For each point, finds its index in the data array and assigns its cluster label 
    # to the corresponding element in the cluster_labels array.
    for i, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[np.all(data == point, axis=1)] = i
    n = len(data) # n is the number of points in the dataset.
    # This array used to store the silhouette score for each point in the dataset.
    silhouette_values = np.zeros(n)
    # iterate over each point in the dataset
    # calculates two values:
    # 1. a, average distance between the current point and all other points in its cluster
    # 2. b, minimum average distance between the current point and all points in any other cluster
    cluster_labels = cluster_labels.astype(int)
    for i in range(n):
        a = np.mean([euclidean_distance(data[i], x) for x in clusters[cluster_labels[i][0]]])
        b = np.inf
        for j in range(len(centroids)):
            if not np.all(j == cluster_labels[i]):
                b = min(b, np.mean([euclidean_distance(data[i], x) for x in clusters[j]]))
        #  silhouette score for the current point calculated as:
        # (b - a) / max(a, b) as per "DAMI23L_mkr_Clustering91.pdf"
        silhouette_values[i] = (b - a) / max(a, b) 
    # return mean silhouette score by calculating the mean of all silhouette scores in the silhouette_values array.
    return np.mean(silhouette_values)
