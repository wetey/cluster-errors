'''
    the python script below finetunes a transformer based model. 
    input: model name, dataset
    output: csv file with the predictions and correct label
'''

#import necessary packages
from torch import cuda
import os
import sys
from utils import get_model_and_tokenizer, get_dataset, get_hf_dataset, compute_metrics
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd

cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def tokenize_function(examples):

    '''
    purpose: convert the text to tokens to then feed to the model

    input: examples: the textual data in the dataset

    output: tokenized text
    '''
    #move tokenization to GPU if available 
    device = 'cuda' if cuda.is_available() else 'cpu'

    #return the tokenized text
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length = 512, return_tensors = 'pt').to(device)

#arguments passed through command line
model_name = sys.argv[1] #model to be fine-tuned. give path on huggingface models
dataset_name = sys.argv[2] #give path to dataset (local or on huggingface)
num_labels = int(sys.argv[3]) #number of labels dataset has
location = sys.argv[4] #location local or hf
output_dir = sys.argv[5] #where to save the output from training

#get model and tokenizer from HF
model, tokenizer = get_model_and_tokenizer(model_name, num_labels)

#get the dataset either from a local file or from huggingface
dataset = get_dataset(dataset_name, split = "train", location = location)

#convert CSV to a format HF model can handle
hf_processed_dataset = get_hf_dataset(dataset)

#create train and test split
hf_processed_dataset = hf_processed_dataset.train_test_split(test_size = 0.15, shuffle = True, seed = 42)
train_set_hf = hf_processed_dataset['train']
test_set_hf = hf_processed_dataset['test']

#batch tokenize train and test sets
tokenized_train_dataset = train_set_hf.map(tokenize_function, batched=True)
tokenized_test_dataset = test_set_hf.map(tokenize_function, batched = True)

#determine the training arguments
training_args = TrainingArguments(output_dir = output_dir,
                                  evaluation_strategy = 'epoch',
                                  logging_steps = 1,
                                  num_train_epochs = 5,
                                  learning_rate = 1e-5,
                                  eval_accumulation_steps = 2)

#create a Trainer object 
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_dataset,
    eval_dataset = tokenized_test_dataset,
    compute_metrics = compute_metrics,
)

#Train the model
results = trainer.train()

#save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#get predictions
predictions, labels, metrics = trainer.predict(tokenized_test_dataset)

#save metrics
trainer.log_metrics('predict', metrics)
trainer.save_metrics('predict', metrics)

#save the predicted label, actual label, and text in a csv file (same directory as trained model)
predictions = np.argmax(predictions, axis=1)
predictions_df = pd.DataFrame()
predictions_df['predicted'] = predictions
predictions_df['label'] = labels
predictions_df['text'] = test_set_hf['text']
predictions_df.to_csv(output_dir+'/predictions.csv', header = True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''