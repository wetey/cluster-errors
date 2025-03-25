# Representing and Clustering Error in Offensive Lanuage Detection

This repository contains all the code and files necessary to reproduce the results from the paper. 

## Project structure

The project is structured as follows:

```txt
.
├── 📂 data/: 
│   └── 📂 clusterings: files generated from running clustering.
│   └── 📂 datasets: the data files generated, original datasets, test sets, and misclassified examples.
│   └── 📂 elbow_plots: elbow plots generated to determine the optimal number of clusters.
│   └── 📂 embeddings: files generated when preparing the text embeddings
├── 📂 human_evaluation/: contains scripts and files used to generate the human evaluation surveys. In addition, the results from the evaluation.
│   └── 📂 scripts: contains the code used to generate the questions for evaluation
│   └── 📂 questions: the questions generated for each embedding type
│   └── 📂 evaluation: evaluation results organized by language
│   └── 📂 figures: generated clustering plots and heatmaps
├── 📂 scripts/: contains scripts used to fine-tune the models.
├── 📂 cluster: scripts to generate the embeddings and run the clustering.
```
More information about the specific files in each directory can be found in the README.md in each directory
## Installation

### conda
We provided the conda environment that contains all the python packages needed to run the program. If you don't have Miniconda/Anaconda installed on your device, you can check [this resource](https://docs.anaconda.com/miniconda/miniconda-install/) on how to install.

### python packages

To create a conda environment that has all the necessary packages, run the following commnad:

```bash
conda env create -f environment.yml
```

The command above will create a conda environment with the name `cluster-errors`. To activate the environment, use the following command:

```bash
conda activate cluster-errors
```

## To Run

To run the entire framework from beginning to end, you will need to run three scripts in the following order
### step 0: [finetune_models.py](scripts/finetune_models.py)
```bash
python finetune_models.py [model_name] [dataset_name] [number_of_labels] [language] [location] [output_dir]  
```
if you already have a fine-tuned model you can skip this step. refer to this [README](scripts/README.md) for more information about each argument.

### step 1: [prepare_embeddings.py](cluster/prepare_embeddings.py)

```bash
python prepare_embeddings.py [file_path] [embedding] [language] [--model]
```
for more information on the constraints and purpose of each argument refer to this [README](cluster/README.md). this script will compute the embeddings and save them to a JSON file in [here](data/clustering/) and generate an elbow plot (saved [here](data/elbow_plots/)). you will use the elbow plot to determine the optimal number of clusters for your data. <br>

### step 2: [cluster.py](cluster/cluster.py)

```bash
python cluster.py [file_path] [num_clusters]
```
for more information refer to this [README](cluster/README.md). after running this script the JSON file will be updated to include two new columns `cluster` and `centroid`. this is the script that will run the kmeans algorithm.


# Cite

