'''
    This script extracts the content column after group the dataset by cluster. This script is used to geenrate the summarization surveys. The extracted information is stored in a JSON file. 
'''

#import necessary packages
import pandas as pd
import sys

dataset_path = sys.argv[1] #where dataset is stored
save_to = sys.argv[2] #where to save the results

clusters = {} #dictionary to store the content column per cluster

#read CSV file
dataset = pd.read_csv(dataset_path)

#group dataset based on cluster column
dataset = dataset.groupby('cluster')

#compute the number of clusters
num_clusters = len(dataset.cluster.unique())

#store the content based on cluster number
for cluster in range(num_clusters):
    clusters[cluster] = list(dataset.get_group(cluster).content)

#save results in dataframe
cluster_number = pd.Series(data = list(clusters.keys()), name = 'cluster_number')
cluster_content = pd.Series(data = clusters.values(), name = 'content')
content = pd.DataFrame({
    cluster_number.name : cluster_number,
    cluster_content.name : cluster_content
})

content.to_json(save_to, indent = 4, orient = 'records', force_ascii = False)