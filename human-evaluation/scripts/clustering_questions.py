'''
    This script generates the questions for clustering task
'''

#load the necessary packages
import pandas as pd
import sys
import random

def pick_four_from_cluster(cluster, cluster_number):

    '''
    purpose: choose four random choices 

    input: cluster: textual examples from cluster
           cluster_number: current cluster number

    output: choice for every cluster and a list of the cluster number (len(current_cluster) == number of questions)
    '''

    choices = [] #2d list that stores the choices
    current_cluster = [] #list of cluster numbers

    #stop when cluster has less than four examples left
    while len(cluster) >= 4:

        #temp list to store the picked choices
        picked = []
        current_cluster.append(int(cluster_number))
        for current in range(4):

            #if there are still examples to choose from
            if(len(cluster) > 0):

                #pick a random index
                index = random.randint(0, len(cluster) - 1)

                #add to temp list
                picked.append(cluster[index])

                #remove from cluster as to not have duplicates
                cluster.remove(cluster[index])

        #append temp list to larger list
        choices.append(picked)
    
    return choices, current_cluster


dataset_path = sys.argv[1] #path to dataset
save_to = sys.argv[2] #path t save results

clusters = {} #store the cluster grouping
cluster_number = [] #list of the cluster each subset belongs to
choices = [] #randomly picked examples
intruder_cluster_list = [] #which cluster intruder belongs to
intruder_cluster_example = [] #randomly picked intruder
number_of_questions = 0

picked_intruders = {} # {"cluster": [(intruder_cluster, example) ...]}
picked_clusters = {} # {"cluster": [choices]}

dataset = pd.read_csv(dataset_path) #read dataset

#group dataset by the cluster
grouped_dataset = dataset.groupby('cluster')

#compute number of clusters
num_clusters = len(dataset.cluster.unique().tolist())

for cluster in range(num_clusters):

    #retrieve the content column for current cluster
    current_list = list(grouped_dataset.get_group(cluster).content)

    #shuffle the list
    random.shuffle(current_list)

    clusters[cluster] = current_list

#pick intruders first
for cluster in range(num_clusters):

    #start with an empty list for every cluster
    picked_intruders[cluster] = []

    #every questions will have four choices from the same cluster
    intruders_needed = len(clusters[cluster]) // 4

    for current in range(intruders_needed):

        #pick a random intruder cluster
        intruder_cluster = random.randint(0, num_clusters - 1)

        #make sure it's not the positive cluster and there are more than 4 examples 
        while intruder_cluster == cluster or len(clusters[intruder_cluster]) <= 4:
            intruder_cluster = random.randint(0, num_clusters - 1)
        
        #pick a random example from intruder cluster
        intruder_index = random.randint(0, len(clusters[intruder_cluster]) - 1)

        #store intruder
        intruder = clusters[intruder_cluster][intruder_index]

        #store as the picked intruder for the current cluster
        picked_intruders[cluster].append((intruder_cluster, intruder))

        #remove intruder from the intruder cluster 
        clusters[intruder_cluster].remove(clusters[intruder_cluster][intruder_index])

#pick four choices at random
for cluster in range(num_clusters):
    current_choices, current_cluster = pick_four_from_cluster(clusters[cluster], num_clusters)
    picked_clusters[cluster] = current_choices

#add intruder
for cluster in range(num_clusters):

    #retrive positive examples and intruders for the current cluster
    current_examples = picked_clusters[cluster]
    current_intruders = picked_intruders[cluster]

    for current_example in range(len(current_examples)):

        #add the intruder to the list of choices
        picked_clusters[cluster][current_example].append(current_intruders[current_example][1])

        #increment number of questions counter
        number_of_questions += 1

        cluster_number.append(int(cluster))

        #store the intruder cluster number
        intruder_cluster_list.append(int(current_intruders[current_example][0]))

        #store the intruder content
        intruder_cluster_example.append(current_intruders[current_example][1])

        #shuffle to avoid having the intruder always be the last choice
        random.shuffle(picked_clusters[cluster][current_example])

    for list in picked_clusters[cluster]:
        choices.append(list)

#store the results in a dataframe
index = range(number_of_questions)
index = [str(value) for value in index]
question = ["question " + str(i) for i in range(number_of_questions)]

index = pd.Series(data = index, name = 'index')
question = pd.Series(data = question, name = 'text')
cluster_number = pd.Series(data = cluster_number, name = 'cluster')
choices = pd.Series(data = choices, name = 'choices')
intruder_cluster_list = pd.Series(data = intruder_cluster_list, name = 'intruder_cluster')
intruder_cluster_example = pd.Series(data = intruder_cluster_example, name = 'intruder_example')

choice_intruder = pd.DataFrame({
    index.name: index,
    question.name: question,
    cluster_number.name: cluster_number,
    choices.name: choices,
    intruder_cluster_list.name: intruder_cluster_list,
    intruder_cluster_example.name: intruder_cluster_example
})

choice_intruder.to_json(save_to, orient = 'records', indent = 4, force_ascii = False)