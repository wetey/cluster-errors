# Cluster

## To run

To generate the embeddings, run [prepare_embeddings.py](prepare_embeddings.py) using the following command:


```bash
python prepare_embeddings.py [file_path] [embedding] [language] [--model]
```

`file_path` = path to file in CSV format. must contain the following columns: content, predicated label (numerical) and label (numerical), predicted label in string format and actual label in string format. and only the misclassified examples. the files used in our paper can be found in the [data directory](../../data) <br>

`embedding` = the embedding types currently supported are: `lhs` (Last Hidden State), `sb` (Sentece-BERT), `lf` (Linguistic Features), `concat` (concatenated lf and sb embeddings) <br>

`language` = language of data to cluster. this program currently only supports English (use en/english) and Arabic (use ar/arabic). <br>

`model` = optional argument if you're using a different model for the embeddings.<br>
- for arabic the default models are: <br>
    - lhs = [wetey/MARBERT-LHSAB](https://huggingface.co/wetey/MARBERT-LHSAB)<br>
    - sb = [sentence-transformers/distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)<br>
- for english the default models are: <br>
    - lhs = [wetey/distilbert-base-uncased-measuring-hate-speech](https://huggingface.co/wetey/distilbert-base-uncased-measuring-hate-speech)<br> 
    - sb = [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)<br>
    
all default models are hosted on huggingface. <br>

The [prepare_embeddings.py](prepare_embeddings.py) script will generate a line plot saved in the [elbow plots directory](../data/elbow_plots/). use the plot to identify the elbow, the value of num clusters at the elbow is your optimal number of clusters. you will need this number to run [cluster.py](cluster.py). In addition, a JSON file is generated which contains the embeddings of the text and is saved in the [clusterings directory](../data/clusterings/).<br>

To run [cluster.py](cluster.py) use the following command:
```bash
python cluster.py [file_path] [num_clusters]
```
`file_path` = path to the JSON file generated from running [prepare_embeddings.py](prepare_embeddings.py). make sure to include the absolute path. <br>

`num_clusters` = the number of clusters at the elbow <br>

## Output

The output of running [prepare_embeddings.py](prepare_embeddings.py) is a JSON file that contains the following information: `content`, `label`, `pred`, `string_label`, `string_pred`, `x` and `y` (for scatter plot visual), and `embedding`. The `lf` JSON file contains two additional columns: `prompts` and `features`. <br>

When you run [cluster.py](cluster.py), the JSON file is updated to include two new columns: `cluster` and `centroid`

The results of clustering are automatically saved in the [clusterings](../data/clusterings) directory.

## Hardware

We used an NVIDIA RTX A6000 GPU to generate all the embeddings. 

`last hidden state` took less than 30 minutes to generate (for both languages).<br>
`Sentence-BERT` took a few seconds to generate (for both languages)<br>
`linguistic features` 
- english features took almost an hour and a half to generate (around 20 minutes to load the model onto the GPU)
- arabic features took a few minutes (these features were generated using [cohere api](https://docs.cohere.com/))

`concatenated` took the same amount of time to generate as `linguistic features`