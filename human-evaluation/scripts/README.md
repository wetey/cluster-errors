# Survey Generation Scripts

## To run

`dataset_path` CSV file can be found in the [questions](questions/) directory, the CSV file in the subdirectories

### [get_cluster_content](get_cluster_content.py)
This script extracts the textual content from all clusters:
```bash
python get_cluster_content.py [dataset_path] [save_to]   
```
`dataset_path` = path to local CSV file </br>
`save_to` = path to store the JSON file generated </br>

### [clustering_questions.py](clustering_questions.py)
This scripts generates the clustering question choices 
```bash
python clustering_questions.py [dataset_path] [save_to]   
```
`dataset_path` = path to local CSV file </br>
`save_to` = path to store the JSON file generated </br>


### [make_survey_clusters.py](make_survey_clusters.py)
This script generate the .txt file in the format Qualtrics accepts. to run use the following command

```bash
python make_survey_clusters.py [questions] [save_to]
```
`questions` = the JSON file generate from running [clustering_questions.py](clustering_questions.py) <br>
`save_to` = path to save the txt file generated. 
