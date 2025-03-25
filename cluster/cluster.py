import cluster.cluster_utils as cluster_utils
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(
                    prog = 'kmeans',
                    description = 'run KMeans clustering')

#path to file which contain the text examples
parser.add_argument('file_path', 
                    help = 'path to file in JSON format. this should be the generated after running cluster.py. to update the same file make sure you are using the absolute path of the JSON file.')

#optimal number of clusters
parser.add_argument('num_clusters',
                    type = int,
                    help = 'enter the number of optimal number of clusters. refer to the elbow plot generated from running cluster.py. the optimal number of clusters is at the elbow.')

#parse arugments passed by user
args = parser.parse_args()

file_path = args.file_path
num_clusters = args.num_cluters

if os.path.isfile(file_path):
    dataset = pd.read_json(file_path)
else:
    #error handling for entering an invalid file_path argument
    print(f'{file_path} is an invalid file\nplease enter a path to your dataset in JSON format.\nexiting program..')
    exit(1)

dataset = cluster_utils.KMeans_clustering(dataset, num_clusters)

#save results
dataset.to_json(file_path, 
                indent=4, 
                orient='records', 
                force_ascii=False) #set to False to the arabic text is human readable in the json file

