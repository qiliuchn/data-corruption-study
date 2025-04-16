# Data corruption study

This repository contains the code and data used in the paper *Navigating Data Corruption in Machine Learning: Balancing Quality, Quantity, and Imputation Strategies*.

Contributors:
Qi Liu and Wanjing Ma

## Overview
### NLP-SL experiment
Folder ```NLP-SL``` contains the code and data for the NLP-SL experiment.

```bert_pretrain_using_wikitext_and_bookcorpus.ipynb``` is for BERT pretraining.

```bert_finetune_on_glue_tasks.py``` is for BERT finetuning on GLUE tasks.

![NLP-SL](assets/data_corruption_effect_nlp_details.png "GLUE tasks")


### Signal-RL experiment
Folder ```Singnal-RL``` contains the code and data for the Signal-RL experiment.

![Signal-RL](assets/sumo_illustration.png "SUMO env illustration")

### results visualization
See the following files for results visualization:

```results_analysis_part1.ipynb```
![Data corruption ratio vs model performance](assets/data_missing_curve_fit.png "Data corruption ratio vs model performance")

```results_analysis_part2.ipynb```
![Imputation advantage heatmap](assets/imputation_advantage_heatmaps.png "Imputation advantage heatmap")

```results_analysis_part3.ipynb```
![enlarging dataset](assets/enlarging_dataset.png "enlarging dataset")