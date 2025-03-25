import numpy as np
from tqdm import tqdm
import pandas as pd 
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import cohen_kappa_score
import collections
import altair as alt
import plotly.express as px


def clustering_plot(dataset, name='N/A'):
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
    
    if name != 'N/A':
        scatter.save(f'figures/{name}_clustering.pdf')
        scatter.save(f'figures/{name}_clustering.svg')
    return scatter 

def majority_label(dataset):

    clusters = dataset.cluster.unique()

    # group dataset based on cluster
    grouped_dataset = dataset.groupby('cluster')

    majority_label = {cluster: None for cluster in clusters}
    count_with_majority_per_cluster = []

    for cluster, data in grouped_dataset:
        labels = data['string_pred'].to_list()

        # Get the majority label per cluster
        majority_label[cluster] = max(set(labels), key=labels.count)

        count_with_majority = len(data[data.string_pred == majority_label[cluster]])
        count_with_majority_per_cluster.append(count_with_majority / len(data))

    # average % of datapoints from the majority label per cluster
    average_count_with_majority = (sum(count_with_majority_per_cluster) / len(count_with_majority_per_cluster)) * 100

    print("Average count_with_majority per cluster:", average_count_with_majority)
    
def display_label_distribution(dataset, title = 'N\A'):
    bar_chart = alt.Chart(dataset).mark_bar().encode(
    x=alt.X("cluster:N").title('Cluster Number'
                               ).axis(labelColor='black', 
                                      labelFontSize=8, 
                                      titleColor='black'),
    y=alt.Y("count(string_pred)").stack("normalize"
                                       ).title('% Predicted Label'
                                               ).axis(labelColor='black', 
                                                      labelFontSize=8,
                                                      titleColor='black'),
    color=alt.Color('string_pred').scale(range=['#5254a3', '#8ca252', '#bd9e39']).title("Predicted Label").legend(titleColor='black', labelColor='black')
    ).properties(
            width = 150,
            height = 150,
            title=alt.Title(text=title)
    )
    return bar_chart

def print_cluster_size(clusters):

    cluster_size = collections.Counter(clusters)
    x_values = list(cluster_size.keys())
    y_values = list(cluster_size.values())   
    categories = pd.DataFrame({
        'cluster':x_values,
        'size': y_values
    })
    base = alt.Chart(categories).encode(
        x = 'cluster:N',
        y = 'size'
    )
    return base.mark_bar()


def get_target_group_distribution(dataset, name='N/A'):
    targets = ['target_race', 
               'target_religion', 
               'target_origin', 
               'target_gender', 
               'target_sexuality',
               'target_age']
    target_labels = [target.split('_')[1] for target in targets]
    grouped_dataset = dataset.groupby('cluster')
    cluster_percentages = {}
    all_labels = set()
    for cluster, data in grouped_dataset:
        total_count = len(data)
        percentages = [(data[target].sum() / total_count) for target in targets]
        cluster_percentages[cluster] = percentages
        all_labels.update(targets) 

    cluster_percentages_df = pd.DataFrame(cluster_percentages, index=target_labels).T

    fig = px.imshow(cluster_percentages_df, 
                    text_auto=".2f",    
                    color_continuous_scale='Purples',
                    zmin=0,
                    zmax=1)
    fig.update_layout(
        showlegend=False,
        xaxis_title="Target Groups",
        yaxis_title="Clusters",
        xaxis=dict(
            tickmode='array',
            tickvals=[i for i in range(len(cluster_percentages_df.columns))],
            ticktext=cluster_percentages_df.columns,
            side='bottom',
            scaleanchor='y',
            scaleratio=1,
            griddash = 'solid',
            showgrid = True
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[i for i in range(len(cluster_percentages_df.index))],
            ticktext=[f"{cluster}" for cluster in cluster_percentages_df.index],
            scaleanchor='x',
            scaleratio=1
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        width = 400,
        height = 400
    )
    fig.update_coloraxes(showscale=False) 
    fig.show()
    if name != 'N/A':
        fig.write_image(f'figures/{name}_targets.pdf', format='pdf')

def add_target_groups(original_dataset, dataset):

    with open('../data/drop_columns.txt') as file:
        columns_to_drop = [line.strip() for line in file.readlines()]

    try: 
        original_dataset = original_dataset.drop(columns_to_drop, axis = 1)
    except:
        pass

    #content column to compare with original dataset
    dataset_content = dataset.content

    #retrieve all columns for each row in test set from original dataset
    subset_original_content = original_dataset.loc[original_dataset.content.isin(dataset_content)]

    #merge dataframes
    dataset = pd.merge(dataset, subset_original_content, on = 'content')
    dataset.drop(dataset.columns[dataset.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
    return dataset

def display_categorical_pie_chart(categories):
    categories = prepare_category_data(categories, 'color', 'theta')

    base = alt.Chart(categories).mark_arc().encode(

        theta = alt.Theta(field = 'theta', type = 'quantitative', stack = True),
        color = alt.Color(field = 'color', type = 'nominal', title = 'category'),
        order = alt.Order(field = 'theta')
    ).properties(
        width = 250,
        height = 250
    )

    pie = base.mark_arc(outerRadius=100).encode(order=alt.Order(field="theta"))

    text = base.mark_text(radius = 115).encode(

        # we just need 1 decimal placae
        text=alt.Text('theta:Q', format='.1f'),

        # show text when theta > 0
        color=alt.condition(
            alt.datum.theta > 0,
            alt.value("black"),
            alt.value(None)
        )
    )   
    return pie + text

def display_categorical_bar_chart(categories):
    categories = prepare_category_data(categories, 'predicted', 'percentage')
    base = alt.Chart(categories).encode(
        x = 'predicted',
        y = 'percentage'
    )
    return base.mark_bar()

def prepare_category_data(categories, label_x, label_y):

    categories = collections.Counter(categories)
    x_values = list(categories.keys())
    y_values = list(categories.values())
    y_values = list(map(lambda y:(y / sum(y_values)*100), y_values))
    categories = pd.DataFrame({
        label_x:x_values,
        label_y: y_values
    })

    return categories

def get_gold_answers(original_questions):
    index_of_intruder = []
    for original in range(len(original_questions)):
        intruder = original_questions.loc[original, 'intruder_example']
        index = original_questions.loc[original, 'choices'].index(intruder)
        index_of_intruder.append(index + 1)
    return index_of_intruder

def get_annotator_answers(evaluation_results):
    answers = []
    for question in evaluation_results:
        answers.extend(evaluation_results[question].values.tolist())
    return answers

def convert_to_np_array(list):

    embeddings = []

    for embedding in list:
        embedding_list = [float(current) for current in embedding.replace('[', '').replace(']', '').split()]
        embeddings.append(embedding_list)

    return np.array(embeddings)

def get_files(evaluation_results_path, original_questions_path):
    #read the necessary files
    evaluation_results = pd.read_csv(evaluation_results_path)
    original_questions = pd.read_json(original_questions_path, orient = 'records')

    return original_questions, evaluation_results

def get_centroids(dataset):
    centroids = []
    for cluster in dataset.cluster.unique():
        centroids.append(dataset[dataset.cluster == cluster].centroid.to_list())

    return centroids

def get_clustering_agreement(annotators):
    annotators = get_clustering_annotations(annotators)
    agreement = np.ones((len(annotators), len(annotators)))
    for annotator_A in range(len(annotators)):
        for annotator_B in range(len(annotators)):
            if annotator_A != annotator_B:
                agreement[annotator_A][annotator_B] = annotator_agreement(annotators[annotator_A], annotators[annotator_B])
    return agreement

def display_clustering_agreement(annotators):
    agreement = get_clustering_agreement(annotators)
    matrix = sn.heatmap(agreement, annot=True, cmap='Purples', fmt=".3f")
    plt.show()

def average_agreement(agreement):
    sum_agreement = 0
    count = 0
    for A in range(len(agreement)):
        for B in range(len(agreement)):
            if A != B and B < A:
                sum_agreement += agreement[A][B]
                count += 1
    return sum_agreement / count, sum_agreement, count

def get_clustering_annotations(annotators_answers):
    answers = []
    for index in  range(3):
        answers.append(annotators_answers.iloc[index].tolist())
    return answers

def annotator_agreement(annotator_A, annotator_B):
    agreement = cohen_kappa_score(annotator_A, annotator_B)
    return agreement

def display_confusion_matrix(flat_total, flat_summary):
    cm = confusion_matrix(flat_total, flat_summary)
    matrix = sn.heatmap((cm/np.sum(cm))*100, annot=True, cmap='Purples', fmt=".2f")
    matrix.set_xlabel('\nannotator answer')
    matrix.set_ylabel('correct answer');
    plt.show()

def clustering_task(original_clustering_questions, clustering_questions):
    index_of_intruder = []
    for original in range(len(original_clustering_questions)):
        intruder = original_clustering_questions.loc[original, 'intruder_example']
        index = original_clustering_questions.loc[original, 'choices'].index(intruder)
        index_of_intruder.append(index + 1)
        
    majority = 0
    for intruder, answer in zip(index_of_intruder, clustering_questions):
        correct = (clustering_questions[answer].values == intruder).sum()
        if correct >= 2:
            majority += 1
    
    print(f'accuracy: {"{:0.2f}".format(majority / len(original_clustering_questions) * 100)}')
