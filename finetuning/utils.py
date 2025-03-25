'''
This script contains helper functions that are used for training, inference, and obtaining sentence-bert embeddings
'''

#import the necessary packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import cuda
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
import torch

def get_dataset(dataset_name, split = None, location = 'hf'):

    '''
    retrieve dataset to use for fine-tuning
    
    input: dataset_name: path where dataset is stored (huggingface or local)
           split: which split to load (train/test/None) default to None
           location: whether dataset is hosted on hugginface (hf) or not (local)

    output: huggingface Dataset object 
    '''

    #if dataset is not available locally, download from HuggingFace 
    if  location == 'hf':
        dataset = load_dataset(dataset_name, split = split)

    #otherwise get data from specified path
    elif location == 'local':
        dataset = load_dataset('csv', data_files = dataset_name, split = split)

    return dataset

def get_hf_dataset(dataset):

    '''
    convert the dataset to a format that can be used for training
    
    input: dataset: dataset used for fine-tuning/training. 
    
    output: huggingface Dataset object 
    '''

    #convert to pandas to replace labels
    processed_dataset = dataset.to_pandas()

    #get the labels
    old_labels = processed_dataset.label.unique()

    #list of numbers 0->len(old_labels) - 1
    new_labels = range(len(old_labels))

    #model requires labels be numerical
    processed_dataset.label = processed_dataset.label.replace(old_labels, new_labels)

    #convert back to HuggingFace dataset
    hf_processed_dataset = Dataset.from_dict(processed_dataset)

    return hf_processed_dataset 

def get_model_and_tokenizer(model_name = 'distilbert-base-uncased', num_labels = 2):

    '''
    get the appropriate model and tokenizer
    
    input: model_name: model path found on huggingface
           num_labels = number of labels in the dataset

    output: HF model and tokenizer
    '''

    #load tokenizer. using autotokenizer so any model can be used
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #load model onto gpu if available for faster training
    device = 'cuda' if cuda.is_available() else 'cpu'

    #load the appropriate classification head
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels).to(device)

    return model, tokenizer

def compute_metrics(eval_pred):

    '''
    compute metrics method required by the transformer train function for model evaluation
    
    input: eval_pred: predictions made by the model

    output: classification metrics: accuracy, precision, recall, and f1 as well as the loss
    '''

    #load the appropriate evaluation metrics
    accuracy_metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    #get the logits and labels
    logits, labels = eval_pred

    #predicted label is that with the highest logit
    predictions = np.argmax(logits, axis=-1)

    #use cross_entropy to calculate the loss
    loss = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()

    #compute the evaluation metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    precision = precision_metric.compute(predictions=predictions, references=labels,average = 'weighted')['precision']
    recall = recall_metric.compute(predictions=predictions, references=labels,average = 'weighted')['recall']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average = 'weighted')['f1']

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'loss':loss}

