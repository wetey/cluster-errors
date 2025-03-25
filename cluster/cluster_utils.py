import altair as alt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import altair as alt
from sklearn import metrics

def clustering_plot(dataset):
    scatter = alt.Chart(dataset).mark_point(size=150, filled=True).encode(

        x = alt.X('x:Q', axis=None), #x-coordinate
        y = alt.Y('y:Q', axis=None), #y-coordinate

        #using a 20 color palette to avoid repeating colors
        color = alt.Color('cluster:N',scale=alt.Scale(scheme='tableau20')
                          #configure the legend
                          ).legend(direction = 'vertical', 
                                   symbolSize = 200, 
                                   labelFontSize=12, 
                                   titleFontSize=14),
        shape = alt.Shape('string_pred:N', 
                          scale=alt.Scale(
                                    range = ['circle', 'diamond', 'square'])
                                    ).legend( direction = 'vertical', 
                                              symbolSize = 200, 
                                              labelFontSize=12, 
                                              titleFontSize=14,
                                              title='predicted'),

        tooltip = ['content:N','cluster:N', 'string_label:N','string_pred:N', 'loss:Q'],

        ).properties(
        width = 600,
        height = 400

        #allows to hover over the dots and read the columns in tooltip
        ).interactive()
    
    return scatter 

def KMeans_clustering(dataset, num_clusters):
    embeddings = np.array(dataset.embedding.tolist())
    Kmeans_clusterer = KMeans(n_clusters = num_clusters,
                              random_state = 42)
    clusters = Kmeans_clusterer.fit_predict(embeddings)
    dataset['cluster'] = pd.Series(clusters, index = dataset.index).astype('int')
    dataset['centroid'] = dataset.cluster.apply(lambda x: Kmeans_clusterer.cluster_centers_[x])
    
    return dataset

def find_optimal_num_clusters(embeddings, name):

    kmeans_metric = []
    kmeans_inertia = []

    k_clusters = np.arange(3,30)

    for n_cluster in tqdm(k_clusters):

        #sklearn implementation
        kmeans = KMeans(n_clusters = n_cluster, random_state = 42)

        #fit model and return labels
        labels = kmeans.fit_predict(embeddings)

        #calculate silhouette score
        metric = metrics.silhouette_score(embeddings, labels, metric='cosine', random_state = 42)

        inertia = kmeans.inertia_

        kmeans_metric.append(metric)
        kmeans_inertia.append(inertia)

    kmeans_metric = pd.Series(data = kmeans_metric, name = 'metric')
    clusters = pd.Series(data = range(3,30), name = 'num clusters')

    kmeans_inertia = pd.Series(data = kmeans_inertia, name = 'inertia')
    results = pd.DataFrame({
        kmeans_metric.name: kmeans_metric,
        clusters.name: clusters,
        kmeans_inertia.name: kmeans_inertia
    })

    chart = alt.Chart(results).mark_line(point = True).encode(
        x = 'num clusters:Q',
        y = 'inertia:Q'
    )
    chart.save(f'{name}.png')