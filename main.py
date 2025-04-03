import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d


def load(path):
    data = np.loadtxt(path, delimiter=';')
    x=data[:,:-1]
    y=data[:,-1]
    return x,y

def k_means(x, y_true):
    #print(x)
    range_n_clusters = range(2, 10)
    #n_clusters = 2
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(x)
        #print(labels)

        score = silhouette_score(x, labels)
        silhouette_scores.append(score)

    #print(silhouette_scores)
    silhouette_scores = np.array(silhouette_scores)
    best_n_clusters = range_n_clusters[silhouette_scores.argmax()]
    worst_n_clusters = range_n_clusters[silhouette_scores.argmin()]

    print(f'Best n_clusters: {best_n_clusters}')
    print(f'Worst n_clusters: {worst_n_clusters}')

    draw(range_n_clusters, silhouette_scores)

def draw(range_n_clusters, silhouette_scores):
    plt.plot(range_n_clusters, silhouette_scores)
    plt.title(f'Experiment: 1 | Method: K-means | Data: {i}_{j}.csv')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()

if __name__ == '__main__':

    for i in range(2, 4):
        for j in range(1, 4):
            X, y_true = load(f'data/{i}_{j}.csv')
            k_means(X, y_true)


    #X, y_true = load('data/2_1.csv')
    #k_means(X, y_true)

