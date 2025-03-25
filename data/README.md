# Data



## [clusterings](clusterings)

This directory contains the JSON files generated from running [cluster.py](../cluster/cluster.py) on the optimal number of clusterings. All files follow the same naming convention `<languagee>`_`<embedding_type>`_clustering.json. 

`language`: ar for arabic and en for english <br>
`embedding_type`: lhs (Last Hidden State), sb (Sentece-BERT), lf (Linguistic Features), concat (concatenated lf and sb embeddings) 

## [datasets](datasets)

This directory includes the datasets we used to fine-tune the models on. These are the dataset after removing duplicates and removing the `Neutral` class and instances with less than three anotations (last two are only for MHS dataset)

### [L-HSAB.csv](L-HSAB.csv)

Each example can belong to one of the following labels: `abusive`, `hate`, and `normal`.

```
@inproceedings{mulki2019hsab, title={L-HSAB: A Levantine Twitter Dataset for Hate Speech and Abusive Language},
author={Mulki, Hala and Haddad, Hatem and Ali, Chedi Bechikh and Alshabani, Halima},
booktitle={Proceedings of the Third Workshop on Abusive Language Online},
pages={111--118},
year={2019}
}

```

### [lhsab_test.csv](lhsab_test.csv)

Test split contains the following columns:

`content` = textual examples <br>
`label` = gold label assigned to the example (numerical) <br>
`pred` = label predicted by the fine-tuned model (numerical) <br>
`string_label` = gold label assigned to the example (string) <br>
`string_pred` = label predicted by the fine-tuned model (string) <br>


### [lhsab_errors.csv](lhsab_errors.csv)

CSV file contains the misclassified examples with the following columns:

`content` = textual examples <br>
`label` = gold label assigned to the example (numerical) <br>
`pred` = label predicted by the fine-tuned model (numerical) <br>
`string_label` = gold label assigned to the example (string) <br>
`string_pred` = label predicted by the fine-tuned model (string) <br>

### [MHS.csv](MHS.csv)
```
@article{kennedy2020constructing,
  title={Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application},
  author={Kennedy, Chris J and Bacon, Geoff and Sahn, Alexander and von Vacano, Claudia},
  journal={arXiv preprint arXiv:2009.10277},
  year={2020}
}
```

### [mhs_test.csv](mhs_test.csv)

Test split contains the following columns:

`content` = textual examples <br>
`label` = gold label assigned to the example (numerical) <br>
`pred` = label predicted by the fine-tuned model (numerical) <br>
`string_label` = gold label assigned to the example (string) <br>
`string_pred` = label predicted by the fine-tuned model (string) <br>


### [mhs_errors.csv](mhs_errors.csv)

CSV file contains the misclassified examples with the following columns:

`content` = textual examples <br>
`label` = gold label assigned to the example (numerical) <br>
`pred` = label predicted by the fine-tuned model (numerical) <br>
`string_label` = gold label assigned to the example (string) <br>
`string_pred` = label predicted by the fine-tuned model (string) <br>


## [embeddings](embeddings)

This directory contains the JSON files generated from running [prepare_embedding.py](../cluster/prepare_embedding.py). All files follow the same naming convention: `<language>`_`<embedding_type>`_embeddings.json. 

`language`: ar for arabic and en for english <br>
`embedding_type`: lhs (Last Hidden State), sb (Sentece-BERT)

## [elbow_plots](elbow_plots)

This directory contains the elbow plots generated to determine the number of clusters to use. Plots' names follow this naming convention: `<language>`_`<embedding_type>`_elbow_plot.png
`language`: ar for arabic and en for english <br>
`embedding_type`: lhs (Last Hidden State), sb (Sentece-BERT), lf (Linguistic Features), concat (concatenated lf and sb embeddings) 
