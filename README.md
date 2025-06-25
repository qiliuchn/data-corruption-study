# Data corruption study

## Introduction
This repository contains the code and results from the paper:
> Liu, Q. and Ma, W., 2025. Navigating Data Corruption in Machine Learning: Balancing Quality, Quantity, and Imputation Strategies. Future Internet, 17(6), p.241.

Contributors:
Qi Liu (liuqi_tj@hotmail.com) and Wanjing Ma

## How to run the experiments
### NLP-SL experiment
Folder ```./NLP-SL/``` contains the code and running instructions for the NLP-SL experiment.

Check ```./NLP-SL/README.md``` for more details.

![NLP-SL](assets/data_corruption_effect_nlp_details.png "GLUE tasks")


### Signal-RL experiment
Folder ```./Singnal-RL/``` contains the code and running instructions the Signal-RL experiment.

Check ```./Signal-RL/README.md``` for more details.

![Signal-RL](assets/sumo_illustration.png "SUMO env illustration")


### Experiment results
Results of the experiments are saved at ```./save/```.
 - ```./save/the_effect_of_data_missing/``` contains the results of Sec 3. Learning with corrupted data; 
 - ```./save/imputation/``` contains the results of Sec 4. Effectiveness of Data Imputation; imputation method = "insert noise";
 - ```./save/imputation_wordvec/``` contains the results of Sec 4. Effectiveness of Data Imputation; imputation method = "wordvec";
 - ```./save/imputation_bert/``` contains the results of Sec 4. Effectiveness of Data Imputation; imputation method = "bert";
 - ```./save/can_enlarged_dataset_compensate/``` contains the results of Sec 5. Effectiveness of Enlarging Dataset.

Saved data include terminal outputs and a summary file. 

Example of terminal outputs:
```
Using corruption mode: miss, corruption ratio: 0.15, imputation mode: none, subset ratio: 1.0
finetuning starts...
Available GPUs: 1
================
current task: cola
loading data...
Dataset size before subset: 8551
Loading a subset of the dataset for fine-tuning with ratio: 1.0
Subset size: 8551
preprocess and encode dataset...

Map (num_proc=8):   0%|          | 0/8551 [00:00<?, ? examples/s]
Map (num_proc=8):   1%|          | 64/8551 [00:00<00:14, 570.28 examples/s]
Map (num_proc=8):  17%|█▋        | 1440/8551 [00:00<00:00, 7390.27 examples/s]
Map (num_proc=8):  35%|███▌      | 3008/8551 [00:00<00:00, 10821.29 examples/s]
Map (num_proc=8):  51%|█████▏    | 4384/8551 [00:00<00:00, 11295.77 examples/s]
Map (num_proc=8):  70%|██████▉   | 5952/8551 [00:00<00:00, 12628.55 examples/s]
Map (num_proc=8):  85%|████████▍ | 7239/8551 [00:00<00:00, 12570.29 examples/s]
Map (num_proc=8):  99%|█████████▉| 8507/8551 [00:00<00:00, 11770.30 examples/s]
Map (num_proc=8): 100%|██████████| 8551/8551 [00:00<00:00, 9842.68 examples/s] 

Map (num_proc=8):   0%|          | 0/1043 [00:00<?, ? examples/s]
Map (num_proc=8):  43%|████▎     | 451/1043 [00:00<00:00, 4084.29 examples/s]
Map (num_proc=8): 100%|██████████| 1043/1043 [00:00<00:00, 4439.00 examples/s]

Map (num_proc=8):   0%|          | 0/1063 [00:00<?, ? examples/s]
Map (num_proc=8):  42%|████▏     | 448/1063 [00:00<00:00, 4442.94 examples/s]
Map (num_proc=8): 100%|██████████| 1063/1063 [00:00<00:00, 4745.75 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at ./google-bert/bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
creating model...
Model size: 109.48M
start training...

  0%|          | 0/1608 [00:00<?, ?it/s]
  0%|          | 1/1608 [00:00<19:02,  1.41it/s]
  0%|          | 2/1608 [00:00<11:59,  2.23it/s]
  0%|          | 3/1608 [00:01<09:27,  2.83it/s]
  0%|          | 4/1608 [00:01<08:31,  3.14it/s]
  0%|          | 5/1608 [00:01<08:11,  3.26it/s]
  0%|          | 6/1608 [00:02<07:53,  3.39it/s]
  0%|          | 7/1608 [00:02<07:36,  3.51it/s]
  0%|          | 8/1608 [00:02<07:22,  3.61it/s]
  1%|          | 9/1608 [00:02<07:14,  3.68it/s]
  1%|          | 10/1608 [00:03<07:08,  3.73it/s]
  1%|          | 11/1608 [00:03<07:02,  3.78it/s]
  1%|          | 12/1608 [00:03<07:11,  3.70it/s]
  1%|          | 13/1608 [00:03<07:10,  3.70it/s]
```


## Visualization
See the following ipython files for results visualization. These scripts will use the results in ```./save/``` to generate figures.

Don't change the name of the folders and files in ```./save/```.

 - ```results_analysis_part1 (NLP-SL).ipynb``` and ```results_analysis_part1 (Signal-RL).ipynb```
![Data corruption ratio vs model performance](assets/data_missing_curve_fit.png "Data corruption ratio vs model performance")

 - ```results_analysis_part2 (NLP-SL).ipynb``` and ```results_analysis_part2 (Signal-RL).ipynb```
![Imputation advantage heatmap](assets/imputation_advantage_heatmaps.png "Imputation advantage heatmap")

 - ```results_analysis_part3 (NLP-SL).ipynb``` and ```results_analysis_part3 (Signal-RL).ipynb```
![enlarging dataset](assets/enlarging_dataset.png "enlarging dataset")

