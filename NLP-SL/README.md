# NLP-SL Experiment
This folder contains the code and running instructions for the NLP-SL experiment.

![NLP-SL](../assets/data_corruption_effect_nlp_details.png "GLUE tasks")


## Model
bert-base-uncased + classification_head

Note: 
Function ```perturb_sentence``` from ```bert_finetune_on_glue_tasks.py``` script is responsible for the data corruption as well as the artificial "inserting noise" imputation.


## Dataset
Pretraining dataset: 
[Wikitext](https://huggingface.co/datasets/Salesforce/wikitext) and [Bookcorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)

Finetuning dataset: [GLUE](https://huggingface.co/datasets/glue)


## Training configurations
### Table 1: NLP-SL Configuration

| Task                     | NLP-SL                                                                                             |
|--------------------------|----------------------------------------------------------------------------------------------------|
| **Model type**           | BERT                                                                                               |
| **Architecture & model description** | - `bert-base-uncased + classification_head`  <br> - Vocab Size: 30,522 (WordPiece)  <br> - `num_layers`: 12  <br> - `num_heads`: 12  <br> - `hidden_att`: 768  <br> - `hidden_ffn`: 3072 |
| **Datasets**             | - Pretraining: Wikitext, Book Corpus  <br> - Finetuning: GLUE                                      |
| **Training config**      | **Pretraining**:  <br> - `batch_size`: 256  <br> - `max_seq_len`: 512  <br> - LR scheduler: linear warmup and decay  <br> - `lr_peak`: 1e-4  <br> - `weight_decay`: 0.01  <br><br> **Finetuning** (CoLA, SST2, MRPC, QQP, MNLI, QNLI, RTE, WNLI):  <br> - `num_epochs`: [5, 3, 5, 16, 3, 3, 3, 6, 2]  <br> - `batch_size`: [32, 64, 16, 16, 256, 256, 128, 8, 4]  <br> - `lr`: [3e-5, 3.5e-5, 3e-5, 3e-5, 5e-5, 5e-5, 5e-5, 2e-5, 1e-5]  <br> - `weight_decay`: 0.01 |


## How to run
 - Run ```bert_pretrain_using_wikitext_and_bookcorpus.ipynb``` for BERT pretraining.


 - Run ```bert_finetune_on_glue_tasks.py``` is for BERT finetuning on GLUE tasks.

Or, you can run ```run_tasks.sh```.