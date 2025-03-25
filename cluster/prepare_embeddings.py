import argparse
import pandas as pd
import os
from embedding import last_hidden_state, sentence_bert, linguistic_features, concatenate_embeddings
import cluster.cluster_utils as cluster_utils

valid_languages = ['ar', 'arabic', 'en', 'english']
valid_embeddings = ['lhs', 'sb', 'lf', 'concat']
lhs_models = {'ar':'wetey/MARBERT-LHSAB',
              'arabic':'wetey/MARBERT-LHSAB',
              'en':'wetey/distilbert-base-uncased-measuring-hate-speech',
              'english':'wetey/distilbert-base-uncased-measuring-hate-speech'
              }
sb_models = {'ar':'sentence-transformers/distiluse-base-multilingual-cased-v1',
              'arabic':'sentence-transformers/distiluse-base-multilingual-cased-v1',
              'en':'sentence-transformers/all-distilroberta-v1',
              'english':'sentence-transformers/all-distilroberta-v1'
              }
large_language_models = {'ar':'command-r',
              'arabic':'command-r',
              'en':'mistralai/Mixtral-8x7B-Instruct-v0.1',
              'english':'mistralai/Mixtral-8x7B-Instruct-v0.1'
              }

parser = argparse.ArgumentParser(
                    prog = 'cluster',
                    description = 'cluster the erroneous examples')

#path to file which contain the text examples
parser.add_argument('file_path', 
                    help = 'path to file in CSV format. must contain the following columns: content, predicated label (numerical) and label (numerical), predicted label in string format and actual label in string format. and only the misclassified examples')

#embedding type [lhs, sb, lf, or concat]
parser.add_argument('embedding', 
                    help = 'embedding type: lhs (Last Hidden State)\nsb (Sentece-BERT)\nlf (Linguistic Features)\nconcat (concatenated lf and sb embeddings)')

#language [ar/arabic or en/english]
parser.add_argument('language', 
                    help = 'language of data to cluster. this program currently only supports English (use en/english) and Arabic (use ar/arabic).')

#path to embedding model (optional argument, default values are what were used in the paper)
parser.add_argument('-','--model', 
                    help = '''optional argument if you\'re using a different model for the embeddings.
                    for arabic the default models are:
                            \'wetey/MARBERT-LHSAB\' for lhs,
                            \'sentence-transformers/distiluse-base-multilingual-cased-v1\' for sb.
                    for english the default models are:
                            \'wetey/distilbert-base-uncased-measuring-hate-speech\' for lhs, 
                            \'sentence-transformers/all-distilroberta-v1\' for sb. 
                    all default models are hosted on huggingface.''',
                    required = False)

#parse arugments passed by user
args = parser.parse_args()

#assign command line arguments to variables for better readibility
embedding = args.embedding
language = args.language
file_path = args.file_path

if os.path.isfile(file_path):
    dataset = pd.read_csv(file_path)
else:
    #error handling for entering an invalid file_path argument
    print(f'{file_path} is an invalid file\nplease enter a path to your dataset in csv form.\nexiting program..')
    exit(1)

#add error handeling for language argument
if language not in valid_languages:
    #error handling for entering invalid language argument
    print(f'{language} is an invalid input.\nvalid options are:\n\tar or arabic if you are using arabic text\n\ten or english if you are using english text.\nexiting program...')
    exit(1)

if embedding not in valid_embeddings:
    #add error handling for entering invalid embedding argument
    print(f'{embedding} is an invalid input.\nvalid options are:\n\tlhs (Last Hidden State)\n\tsb (sentence_BERT)\n\tlf (linguistic features)\n\tconcat (sentence_BERT and linguistic features embeddings concatenated)\nexiting program...')
    exit(1)

if embedding == 'lhs':
    embedding_model = lhs_models[language]
    dataset = last_hidden_state(embedding_model, dataset)
elif embedding == 'sb':
    embedding_model = sb_models[language]
    dataset = sentence_bert(embedding_model, dataset)
elif embedding == 'lf':
    lf_model = large_language_models[language]
    embedding_model = sb_models[language]
    dataset = linguistic_features(lf_model, embedding_model, dataset, language)
elif embedding == 'concat':
    embedding_model = sb_models[language]
    lf_model = large_language_models[language]
    dataset = concatenate_embeddings(lf_model, embedding_model, dataset, language)


#determine the optimal number of clusters
cluster_utils.find_optimal_num_clusters(dataset.embedding)

#save results
dataset.to_json(f'../../data/clusterings/{language}_{embedding}_clustering.json', 
                indent=4, 
                orient='records', 
                force_ascii=False) #set to False to the arabic text is human readable in the json file