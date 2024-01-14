import tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kmeans import k_means, elbow_method, silhouette_scores, plot_clusters_2D, compute_silhouette_scores

data = None
y_true = None

def find_optimal_k():
    global data
    global y_true
    data, y_true = open_dataset()
    print(f'Shape of data: {data.shape}')
    elbow_method(data, max_k=10, max_iterations=100)
    silhouette_scores(data, max_k=10, max_iterations=100)

def open_dataset():
    file_path = filedialog.askopenfilename(
        filetypes=[('ARFF Files', '*.arff')])
    with open(file_path, 'r') as f:
        data_with_labels = arff.loadarff(f)
        df = pd.DataFrame(data_with_labels[0])
        # The true labels are in the last column of the DataFrame
        y_true = df.iloc[:, -1].values
        y_true = y_true.astype(int)  
        # Select only columns containing x and y features i.e w/o lables
        df = df[['x', 'y']]
        #print(f'df: {df}')
        data = np.array(df)
        return data, y_true

def purity_score(y_true, y_pred):
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def calculate_clusters():
    k = int(k_entry.get())
    clusters, centroids = k_means(k, data)
    result_centroids.delete(1.0, tk.END)
    result_clusters.delete(1.0, tk.END)
    result_centroids.insert(tk.END, f"Centroids: {centroids}")
    result_clusters.insert(tk.END, f"Clusters: {clusters}")

    plot_clusters_2D(clusters, centroids)

    # Calculate silhouette score for specific k value
    score = compute_silhouette_scores(data, clusters, centroids)
    result_silhouette.delete(1.0, tk.END)
    result_silhouette.insert(tk.END, f'Silhouette score for k={k}: {score}')

    # Calculate purity
    y_pred = np.empty_like(y_true)
    for i, cluster in enumerate(clusters):
        for point in cluster:
            y_pred[np.all(data == point, axis=1)] = i
    print(f'y_true: {y_true}')
    purity = purity_score(y_true, y_pred)
    print(f'Purity: {purity}')


app = tk.Tk()
app.title("K-Means Clustering")

mainframe = ttk.Frame(app, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

k_label = ttk.Label(mainframe, text="Enter the value of K:")
k_label.grid(column=1, row=1, sticky=tk.W)

k_entry = ttk.Entry(mainframe)
k_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

find_k_button = ttk.Button(
    mainframe, text="Find Optimal K", command=find_optimal_k)
find_k_button.grid(column=3, row=2, sticky=tk.W)

calculate_button = ttk.Button(
    mainframe, text="Calculate Clusters", command=calculate_clusters)
calculate_button.grid(column=3, row=1, sticky=tk.W)

result_label = ttk.Label(mainframe, text="Results:")
result_label.grid(column=1, row=2, sticky=tk.W)

result_centroids = tk.Text(mainframe, wrap=tk.WORD, height=5, width=50)
result_centroids.grid(column=1, row=3, padx=10, pady=10,
                      sticky=(tk.W, tk.E), columnspan=3)

result_clusters = tk.Text(mainframe, wrap=tk.WORD, height=10, width=50)
result_clusters.grid(column=1, row=4, padx=10, pady=10,
                     sticky=(tk.W, tk.E), columnspan=3)

result_label = ttk.Label(mainframe, text="Results:")
result_label.grid(column=1, row=2, sticky=tk.W)

result_centroids = tk.Text(mainframe, wrap=tk.WORD, height=5, width=50)
result_centroids.grid(column=1, row=3, padx=10, pady=10,
                      sticky=(tk.W, tk.E), columnspan=3)

result_clusters = tk.Text(mainframe, wrap=tk.WORD, height=10, width=50)
result_clusters.grid(column=1, row=4, padx=10, pady=10,
                     sticky=(tk.W, tk.E), columnspan=3)

result_silhouette_label = ttk.Label(mainframe, text="Silhouette Score:")
result_silhouette_label.grid(column=1, row=5, sticky=tk.W)

result_silhouette = tk.Text(mainframe, wrap=tk.WORD, height=2, width=50)
result_silhouette.grid(column=1, row=6, padx=10, pady=10,
                       sticky=(tk.W, tk.E), columnspan=3)

app.mainloop()
