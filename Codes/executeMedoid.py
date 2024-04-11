import argparse
import csv
import numpy as np
import os
import pandas as pd
import pygad
from sklearn.metrics import silhouette_score, davies_bouldin_score
from time import perf_counter

from clustering_utils import *
from utils_compare import *
from utils_parser import *



# python Codes/executeMedoid.py Dim032
def main() -> None:
    parser = argparse.ArgumentParser(description='Script for executing medoid clustering.')


    parser.add_argument('data', metavar='data', type=str, choices=DIM_DATA,
                    help='The data to cluster.')

    arguments = parser.parse_args()

    data_file = vars(arguments)['data']
    
    num_generations=81
    sol_per_pop=62
    num_parents_mating=33

    data_dir = (os.path.join(os.path.curdir,'PreparedArtificialData'))

    Datasets_dir = os.path.join(data_dir,'scaled_datasets')

    dataset_path = os.path.join(Datasets_dir, data_file)


    # Read data file

    global X
    X = pd.read_csv(f"{dataset_path}.csv", header = None)

    X = X.to_numpy()

    

    print(f'Dataset {data_file} read\n')

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
    num_genes = num_clusters

    initial_pop = [list(cluster_gonzalezMedoid(X, num_clusters))]
    rng = np.random.default_rng()
    for i in range(sol_per_pop-1):
        initial_pop.append(list(rng.choice(X.shape[0], num_clusters, replace=False)))
    



    ga_instance = pygad.GA(num_generations=int(num_generations),
                       sol_per_pop=int(sol_per_pop),
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_pop,
                       gene_type=int,
                       # init_range_low=0,
                       # init_range_high=X.shape[0]-1,
                       random_mutation_min_val=0,
                       random_mutation_max_val=X.shape[0],
                       mutation_by_replacement=True,
                       allow_duplicate_genes=False,                       
                       gene_space = {"low": 0, "high": X.shape[0]-1},
                       mutation_type='random',
                       num_genes=int(num_genes),
                       fitness_func=fitness_func_DBI,
                       suppress_warnings=True,
                       stop_criteria='saturate_30',
                       on_generation=on_generation,
                       save_best_solutions=True)

    print('CLustering start\n')

    t_start = perf_counter() 
    ga_instance.run()
    t_end = perf_counter()

    print('CLustering end\n')


    time = t_end - t_start
    time = '{:.4f}'.format(time)
    # best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    best_solution = ga_instance.best_solutions[-1]
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, 0)
    
    DBI = davies_bouldin_score(X, cluster_indices)
    sil = silhouette_score(X,cluster_indices)
    sil = '{:.4f}'.format(sil)
    DBI = '{:.4f}'.format(DBI)

    if data_file not in NO_LABELS:
        real_labels_path = os.path.join('Labels',f'{data_file}.txt')
        with open(real_labels_path, 'r') as f:
            real_labels_list = f.readline()
        real_labels_list = real_labels_list.removeprefix('[')
        real_labels_list = ' ' + real_labels_list
        real_labels_list = real_labels_list.removesuffix(']')
        real_labels_array = np.array(real_labels_list.split(','))

        jaccard = calculate_jaccard(real_labels_array, cluster_indices)
        jaccard = '{:.4f}'.format(jaccard)
    else:
        jaccard = None



    print(f"Time: {time}\tSil: {sil}\tDBI: {DBI}\tJAC: {jaccard}")



    
    # best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    best_solution = ga_instance.best_solutions[-1]
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, 0)
    
    plot_clusters(X,cluster_indices,dim=(0,1), points = cluster_centers)

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
    




def on_generation(ga_instance):
    if ga_instance.generations_completed %5 == 0:
        print(f'{ga_instance.generations_completed} generations executed\n')


def cluster_data_DBI(solution, solution_idx):
    global num_clusters, X
    cluster_centers = []
    all_clusters_dists = []

    for i, medoid in enumerate(solution):

        cluster_centers.append(X[medoid])
        cluster_center_dists = euclidean_distance(X, cluster_centers[i])
        all_clusters_dists.append(np.array(cluster_center_dists))

    cluster_centers = np.array(cluster_centers)
    all_clusters_dists = np.array(all_clusters_dists)

    cluster_indices = np.argmin(all_clusters_dists, axis=0)

    return cluster_indices


def cluster_data(solution, solution_idx):
    global num_clusters, X
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []


    for i, medoid in enumerate(solution):

        
        cluster_centers.append(X[medoid])
        cluster_center_dists = euclidean_distance(X, cluster_centers[i])
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

    DBI = davies_bouldin_score(X, cluster_indices)

    fitness = 1.0 / (DBI + 0.00000001)

    return fitness


def euclidean_distance(X, Y):
    return np.sqrt(np.sum(np.power(X - Y, 2), axis=1))

if __name__ == '__main__':
    main()