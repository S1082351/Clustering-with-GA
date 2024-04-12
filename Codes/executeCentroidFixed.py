import argparse
import csv
import numpy as np
import os
import pandas as pd
import pygad
from sklearn.metrics import silhouette_score, davies_bouldin_score

from clustering_utils import *
from utils_compare import *
from utils_parser import *



# python Codes/executeCentroidFixed.py Dim032
def main() -> None:
    parser = argparse.ArgumentParser(description='Script for executing fixed centroid clustering.')

    parser.add_argument('data', metavar='data', type=str, choices=DIM_DATA,
                    help='The data to cluster.')

    arguments = parser.parse_args()

    data_file = vars(arguments)['data']

    random_mutation_max_val = 0.55

    num_generations=66
    sol_per_pop=76
    num_parents_mating=72

    data_dir = (os.path.join(os.path.curdir,'PreparedArtificialData'))
    Datasets_dir = os.path.join(data_dir,'scaled_datasets')

    dataset_path = os.path.join(Datasets_dir, data_file)



    # Read data file

    global X
    X = pd.read_csv(f"{dataset_path}.csv", header = None)

    X = X.to_numpy()

    n_clusters_file = 'Codes/n_clusters_artificial.csv'



    #Reading the number of clusters

    with open(n_clusters_file, 'r') as f:
        # Create a reader object
        reader = csv.DictReader(f)

        # Iterate through the rows
        for row in reader:
            n_clusters_dic = row
    
    global num_clusters
    num_clusters = int(n_clusters_dic[data_file])
    num_genes = num_clusters * X.shape[1]

    # Gonzalez chromosome
    B = cluster_gonzalez2(X, num_clusters)

    labels = np.zeros(X.shape[0], dtype=int)
    for label in range(len(B)):
        for i in B[label]:
            labels[i]=label
    cluster_centers = []
    for cluster in range(num_clusters):
        positions = np.where(labels==cluster)[0]
        cluster_center = np.sum(X[positions],axis=0)/len(positions)
        cluster_centers = np.append(cluster_centers,cluster_center)
    initial_pop = [cluster_centers]

    # Checking DBI and silhouette for the Gonzalez chromosome
    cluster_indices = cluster_data_DBI(cluster_centers, 0)
    
    DBI = davies_bouldin_score(X, cluster_indices)
    sil = silhouette_score(X,cluster_indices)
    sil = '{:.4f}'.format(sil)
    DBI = '{:.4f}'.format(DBI)


    print(f'Initial DBI: {DBI}')
    print(f' Initial sil: {sil}')       

    
    initial_pop = [list(np.zeros(num_genes))]


    
    for i in range(sol_per_pop-1):
        initial_pop.append(list(np.random.uniform(low=0, high=1, size=(num_genes,))))




# Running the GA

    ga_instance = pygad.GA(num_generations=int(num_generations),
                       sol_per_pop=int(sol_per_pop),
                       num_parents_mating=num_parents_mating,
                       gene_type=float,
                       # init_range_low=0,
                       # init_range_high=1,                       
                       random_mutation_max_val=random_mutation_max_val,
                       random_mutation_min_val=-random_mutation_max_val,
                       # keep_parents=1,
                       mutation_type='random',
                       num_genes=int(num_genes),
                       fitness_func=fitness_func_DBI,
                       suppress_warnings=True,                       
                       gene_space = {"low": 0, "high": 1},
                       on_generation=on_generation,
                       stop_criteria='saturate_30',
                       save_best_solutions=True)



    ga_instance.run()







    # best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    best_solution = ga_instance.best_solutions[-1]
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, 0)

    real_centers = []
    for cluster_idx in range(num_clusters):
        cluster_x = X[clusters[cluster_idx], 0]
        cluster_y = X[clusters[cluster_idx], 1]
        if cluster_x.size > 0:
            cluster_center_x = sum(cluster_x)/cluster_x.size
            cluster_center_y = sum(cluster_y)/cluster_y.size
            real_centers.append([cluster_center_x, cluster_center_y])
    real_centers_array = np.array(real_centers)

    plot_clusters(X,cluster_indices,dim=(0,1), points = real_centers_array)





    DBI = davies_bouldin_score(X, cluster_indices)
    sil = silhouette_score(X,cluster_indices)
    sil = '{:.4f}'.format(sil)
    DBI = '{:.4f}'.format(DBI)


    print(f'Final DBI: {DBI}')
    print(f'Final sil: {sil}')



def on_generation(ga_instance):
    if ga_instance.generations_completed %5 == 0:
        print(f'{ga_instance.generations_completed} generations executed\n')


def cluster_data_DBI(solution, solution_idx):
    global num_clusters, X
    feature_vector_length = X.shape[1]
    cluster_centers = []
    all_clusters_dists = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(X, cluster_centers[clust_idx])
        all_clusters_dists.append(np.array(cluster_center_dists))

    cluster_centers = np.array(cluster_centers)
    all_clusters_dists = np.array(all_clusters_dists)

    cluster_indices = np.argmin(all_clusters_dists, axis=0)

    return cluster_indices


def cluster_data(solution, solution_idx):
    global num_clusters, X
    feature_vector_length = X.shape[1]
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(X, cluster_centers[clust_idx])
        all_clusters_dists.append(np.array(cluster_center_dists))

    cluster_centers = np.array(cluster_centers)
    all_clusters_dists = np.array(all_clusters_dists)

    cluster_indices = np.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_clusters):
        clusters.append(np.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = np.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist



def fitness_func_DBI(GA, solution, solution_idx):
    
    cluster_indices = cluster_data_DBI(solution, solution_idx)
    if np.unique(cluster_indices).size < 2:
        fitness = 0
    else:

        DBI = davies_bouldin_score(X, cluster_indices)

        fitness = 1.0 / (DBI + 0.00000001)

    return fitness


def euclidean_distance(X, Y):
    return np.sqrt(np.sum(np.power(X - Y, 2), axis=1))

if __name__ == '__main__':
    main()