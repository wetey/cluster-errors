# Human Evaluation

This directory includes all the files related to the human evaluation we conducted

## [questions](questions/)

Contains all the file generated for human evaluation by running [clustering_questions.py](human-evaluation/scripts/clustering_questions.py). <br>
All json files follow the same naming convention: `<dataset_name>`_`<embedding_type>`_clustering.json


## [evaluation](evaluation/)

Similar to the [questions](questions/) directory, this directory is organized by languages. Evaluation results for Arabic can be found in [ar](results/ar/) and results for English can be found in [en](results/en/).


## [scripts](scripts/)

This directory contains scripts used to generate the surveys in a format that Qualtrics accepts. For information on how to generate the survey, refer to this [README](scripts/README.md)

## [human_evaluation.ipynb](human_evaluation.ipynb)

This notebook contains the human evaluation analysis for English and Arabic for all approaches. 

## [evaluation.utils.py](evaluation.utils.py)

THis python script is not meant to be run. The methods defined are called in [human_evaluation.ipynb](human_evaluation.ipynb)
