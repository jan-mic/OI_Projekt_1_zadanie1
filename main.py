import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
#from matplotlib.patches import Polygon
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, box


def load(path):
    data = np.loadtxt(path, delimiter=';')
    x=data[:,:-1]
    y=data[:,-1]
    return x,y

def k_means(x, y_true, file_i, file_j):
    #print(x)
    range_n_clusters = range(2, 10)
    #n_clusters = 2
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(x)
        #print(labels)

        score = silhouette_score(x, y_pred)
        silhouette_scores.append(score)

    #print(silhouette_scores)
    silhouette_scores = np.array(silhouette_scores)
    best_n_clusters = range_n_clusters[silhouette_scores.argmax()]
    best_silhouette_score = silhouette_scores.max().round(3)
    worst_n_clusters = range_n_clusters[silhouette_scores.argmin()]
    worst_silhouette_score = silhouette_scores.min().round(3)


    info_best = f'Silhouette score: {best_silhouette_score} (best) | N_clusters: {best_n_clusters}'
    info_worst = f'Silhouette score: {worst_silhouette_score} (worst) | N_clusters: {worst_n_clusters}'

    draw(range_n_clusters, silhouette_scores, file_i, file_j)

    kmeans = KMeans(n_clusters=best_n_clusters)
    y_pred = kmeans.fit_predict(x)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info_best)

    kmeans = KMeans(n_clusters=worst_n_clusters)
    y_pred = kmeans.fit_predict(x)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info_worst)

def draw(range_n_clusters, silhouette_scores, file_i, file_j):
    plt.plot(range_n_clusters, silhouette_scores)
    plt.title(f'Experiment: 1 | Method: K-means | Data: {file_i}_{file_j}.csv')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()

def dbscan_experiment(X, y_true, file_i, file_j):
    silhouette_scores = []
    n_clusters_list = []

    eps_values= np.arange(0.01, 2.1, 0.1)

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=1)  # min_samples przykładowo 5
        labels = dbscan.fit_predict(X)

        # Sprawdzamy, ile jest unikalnych etykiet poza -1 (punktami uznanymi za szum)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        n_clusters = len(unique_labels)

        # Obliczamy silhouette_score tylko, jeśli n_clusters > 1
        if n_clusters > 1:
            score = silhouette_score(X, labels)
        else:
            score = 0  # lub np. np.nan, w zależności od preferencji

        silhouette_scores.append(score)
        n_clusters_list.append(n_clusters)

    # Wyszukiwanie najlepszego i najgorszego wyniku
    best_index = np.argmax(silhouette_scores)
    worst_index = np.argmin(silhouette_scores)
    best_eps = eps_values[best_index]
    worst_eps = eps_values[worst_index]

    info_best = f"Najlepszy eps: {best_eps}, silhouette: {silhouette_scores[best_index]:.3f}, liczba klastrów: {n_clusters_list[best_index]}"
    info_worst = f"Najgorszy eps: {worst_eps}, silhouette: {silhouette_scores[worst_index]:.3f}, liczba klastrów: {n_clusters_list[worst_index]}"

    # Rysowanie wykresu zależności silhouette score i liczby klastrów od eps
    # (Tę część można zrealizować np. w matplotlib, analogicznie do K-means)
    #draw(eps_values, silhouette_scores, file_i, file_j)
    plt.plot(eps_values, silhouette_scores)

    for eps in eps_values:
        plt.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)

    #_, y_max = plt.ylim()
    # Dodanie liczby klastrów pod osią x
    for i, eps in enumerate(eps_values):
        plt.text(eps, 0.1,
                 str(n_clusters_list[i]),
                 fontsize=8,  # Rozmiar czcionki
                 transform = plt.gca().get_xaxis_transform())



    plt.title(f'Experiment: 1 | Method: DBSCAN | Data: {file_i}_{file_j}.csv')
    plt.xlabel('Eps')
    plt.ylabel('Silhouette score')
    plt.legend(['silhouette'])
    plt.show()

    # Wizualizacja Woronoja dla najlepszego eps
    dbscan_best = DBSCAN(eps=best_eps, min_samples=1)
    y_pred = dbscan_best.fit_predict(X)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info_best)

    # Wizualizacja Woronoja dla najgorszego eps
    dbscan_worst = DBSCAN(eps=worst_eps, min_samples=1)
    y_pred = dbscan_worst.fit_predict(X)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info_worst)


def plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info):

    # 1. Określ granice wykresu
    min_x, min_y = X.min(axis=0)
    max_x, max_y = X.max(axis=0)
    max_point_value=max(abs(min_x), abs(min_y), abs(max_x), abs(max_y))
    # 2. Dodaj punkty strażników (tworzące "ramkę" wokół danych)
    bounding_frame = np.array([
        [min_x - max_point_value, min_y - max_point_value],  # lewy dolny
        [min_x - max_point_value, max_y + max_point_value],  # lewy górny
        [max_x + max_point_value, min_y - max_point_value],  # prawy dolny
        [max_x + max_point_value, max_y + max_point_value]   # prawy górny
    ])

    X_with_bounding = np.vstack([X, bounding_frame])
    vor = Voronoi(X_with_bounding)
    y_pred=y_pred.astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['orange', 'green', 'blue', 'red', 'yellow', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive']

    cmap = ListedColormap(colors)

    for index, point_region_id in enumerate(vor.point_region):
        region = vor.regions[point_region_id]
        if not -1 in region:  # Ignoruj nieskończone regiony
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=cmap(y_pred[index]), alpha=0.4, edgecolor='k')

    ax.scatter(X[:,0], X[:,1], c=y_true, cmap=cmap, edgecolors='k', s=25)

    # Ustawienia wykresu
    padding = 0.5
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    plt.title(f"Data: {file_i}_{file_j}.csv | {info}", fontsize=10)
    plt.show()


if __name__ == '__main__':

    for i in range(2, 4):
        for j in range(1, 4):
            X, y_true = load(f'data/{i}_{j}.csv')
            #k_means(X, y_true, i, j)
            dbscan_experiment(X, y_true, i, j)

    # X, y_true = load(f'data/2_1.csv')
    # k_means(X, y_true, 2, 1)

    #X, y_true = load('data/2_1.csv')
    #k_means(X, y_true)

