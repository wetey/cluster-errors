# Fine-tuning

This directory contains the scripts we wrote to fine-tune the models. the model we chose to fine-tune for english is the [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) on the [measuring-hates-speech dataset](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech). for arabic we fine-tuned the [UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT) on the [L-HSAB dataset](https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset). the datasets are in the [datasets directory](../data/datasets) (the dataset files we include in this repo are after the cleaning we did which we mention in the paper). <br>

The fine-tuned models can be found on huggingface:
- for english: [wetey/distilbert-base-uncased-measuring-hate-speech](https://huggingface.co/wetey/distilbert-base-uncased-measuring-hate-speech)
- for arabic: [wetey/MARBERT-LHSAB](https://huggingface.co/wetey/MARBERT-LHSAB)

All the fine-tuning was done using an NVIDIA RTX A6000 GPU. It took less than 30 minutes to fine-tune each model.

## To run

### [finetune_models.py](finetune_models.py)
This script prepares the data and does the fine-tuning, to run use the command below:
```bash
python finetune_models.py [model_name] [dataset_name] [number_of_labels] [language] [location] [output_dir]  
```
`model_name` = model name on HuggingFace </br>
`dataset_name` = path to local file or HuggingFace dataset </br>
`number_of_labels` = number of labels in dataset </br>
`location` = 'hf if dataset is from huggingface 'local' if dataset is from file stored </br>
`output_dir` = name of output directory to save checkpoints during fine-tuning </br>

## [utils.py](utils.py)

This python script is not meant to be run. It contains the helper functions used in the fine-tuning process.


