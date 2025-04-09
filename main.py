import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d


def load(path):
    data = np.loadtxt(path, delimiter=';')
    x=data[:,:-1]
    y=data[:,-1]
    return x,y

def k_means(x, y_true, i, j):
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


    info_best_n_clusters = f'Best n_clusters: {best_n_clusters}'
    info_worst_n_clusters = f'Worst n_clusters: {worst_n_clusters}'
    print(f'Best n_clusters: {best_n_clusters}')
    print(f'Worst n_clusters: {worst_n_clusters}')

    draw(range_n_clusters, silhouette_scores, i, j)

    kmeans = KMeans(n_clusters=best_n_clusters)
    #labels = kmeans.fit(x)
    #y_pred = kmeans.labels_.astype(int)
    labels = kmeans.fit_predict(x)
    # Najlepszy przypadek (najwyższy silhouette score)
    kmeans = KMeans(n_clusters=best_n_clusters)
    labels = kmeans.fit_predict(x)

    plot_voronoi_kmeans(X, y_true, labels, best_n_clusters, i, j)
    plot_voronoi_kmeans(X, y_true, labels, worst_n_clusters, i, j)

    # Najgorszy przypadek (najniższy silhouette score)
    #plot_voronoi_kmeans(X, worst_n_clusters, title=f"K-Means, n_clusters={worst_n_clusters} (najgorszy)")

def draw(range_n_clusters, silhouette_scores, i, j):
    plt.plot(range_n_clusters, silhouette_scores)
    plt.title(f'Experiment: 1 | Method: K-means | Data: {i}_{j}.csv')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


# 2. Diagram Woronoja dla najlepszego i najgorszego przypadku
def plot_voronoi_kmeans(X, y_true, y_pred, n_clusters, i, j):
    # Stworzenie diagramu Woronoja

    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(X)

    vor = Voronoi(X)

    # Zdefiniowanie kolorów dla diagramu (pomarańczowy, zielony, niebieski)
    #cmap = plt.colormaps.get_cmap("rainbow")
    colors = ['orange', 'green', 'blue', 'red', 'yellow', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive']

    cmap = ListedColormap(colors)

    # Rysowanie diagramu Woronoja z wypełnianiem komórek
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', show_points=False)

    # Przypisanie kolorów do punktów na podstawie ich etykiety (y_pred)
    for region_idx, point_idx in enumerate(vor.point_region):
        region = vor.regions[point_idx]
        if not -1 in region:  # Ignoruj nieskończone regiony
            polygon = [vor.vertices[i] for i in region]
            #cluster_color = colors[y_pred[point_idx]]
            ax.fill(*zip(*polygon), color=cmap(y_pred[region_idx]), alpha=0.4)

    # Rysowanie punktów
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, s=50, edgecolor='k')

    # Dodatkowo oznacz prawdziwe klasy przezroczystymi okręgami (jeśli y_true != None)
    # if y_true is not None:
    #     ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, s=100, edgecolor='k', alpha=0.5, label='True labels')

    plt.title(f"data | Data: {i}_{j}.csv'")
    plt.grid(True)
    #plt.axis('equal')
    plt.show()


if __name__ == '__main__':

    for i in range(2, 4):
        for j in range(1, 4):
            X, y_true = load(f'data/{i}_{j}.csv')
            k_means(X, y_true, i, j)


    #X, y_true = load('data/2_1.csv')
    #k_means(X, y_true)

