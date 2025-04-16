#===================parse arguments======================
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune BERT model")
parser.add_argument("--corruption_mode", type=str, choices=["none", "miss", "noise"], required=True, help="Corruption mode")
parser.add_argument("--corruption_ratio", type=float, required=True, help="Corruption ratio")
parser.add_argument("--imputation_mode", type=str, choices=["none", "wordvec", "bert", "insert_noise"], required=True, help="Imputation mode")
parser.add_argument("--imputation_noise_level", type=float, default=0.0, help="Imputation mode being insert_noise, the level of noise to be inserted")
parser.add_argument("--subset_ratio", type=float, required=True, help="The size of subset used for finetuning")
args = parser.parse_args()
# Use the arguments
corruption_mode = args.corruption_mode
corruption_ratio = args.corruption_ratio
imputation_mode = args.imputation_mode
imputation_noise_level = args.imputation_noise_level
subset_ratio = args.subset_ratio
print(f"Using corruption mode: {corruption_mode}, corruption ratio: {corruption_ratio}, imputation mode: {imputation_mode}, imputation_noise_level: {imputation_noise_level}, subset ratio: {subset_ratio}")




#===================finetune script======================
print('finetuning starts...')
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
from datasets import load_dataset
import evaluate
import pandas as pd
import numpy as np
import random
from gensim.models import KeyedVectors

import warnings
import logging
# Set logging level to ERROR to suppress warnings in IPython
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import os
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# List of GLUE tasks
glue_tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
'''
	1.	CoLA (Corpus of Linguistic Acceptability): A binary classification task to predict whether a sentence is grammatically acceptable or not. Evaluation metric: Matthews Correlation Coefficient (MCC).
	2.	SST-2 (Stanford Sentiment Treebank): A binary classification task for sentiment analysis, where the model must predict if a sentence has a positive or negative sentiment. Evaluation metric: Accuracy.
	3.	MRPC (Microsoft Research Paraphrase Corpus): A binary classification task where the model determines if two sentences are paraphrases of each other (i.e., if they have the same meaning). Evaluation metric: Accuracy and F1 score.
	4.	STS-B (Semantic Textual Similarity Benchmark): A regression task to predict the similarity score between two sentences on a continuous scale from 0 (completely dissimilar) to 5 (equivalent in meaning). Evaluation metrics: Pearson Correlation and Spearman Correlation.
	5.	QQP (Quora Question Pairs): A binary classification task to determine if two questions are semantically similar (i.e., if they essentially ask the same thing). Evaluation metrics: Accuracy and F1 score.
	6.	MNLI (Multi-Genre Natural Language Inference): A three-class classification task where the model predicts if a hypothesis is entailed, contradicted, or neutral with respect to a given premise. The dataset includes two validation sets: matched (in-domain) and mismatched (out-of-domain). Evaluation metric: Accuracy.
	7.	QNLI (Question Natural Language Inference): A binary classification task derived from the Stanford Question Answering Dataset (SQuAD). The model must determine if a given sentence (hypothesis) is a valid answer to a question (premise). Evaluation metric: Accuracy.
	8.	RTE (Recognizing Textual Entailment): A binary classification task to determine if a hypothesis can be logically inferred from a premise. Evaluation metric: Accuracy.
	9.	WNLI (Winograd Natural Language Inference): A binary classification task based on pronoun resolution. The model determines whether a pronoun in a sentence refers to a specific noun. Evaluation metric: Accuracy.
'''
#glue_tasks = ['rte']

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)

# =====================settings===============================
# Model name
base_model_path = "./bert"  # use BERT pretrained on wikitext and Book-Corpus data
#dataset_path = 'glue'
dataset_path = './nyu-mll/glue'
#corruption_mode = 'miss'  # 'miss' or 'noise' or None
#corruption_ratio = 0.4
# The number of epochs for each task can be adjusted based on the dataset size and task complexity
# tested performances:
# task: bert_base_uncased, self-trained
# cola (MCC): 0.602
# sst2: 0.926
# mrpc: 0.873
# stsb (Pearson): 0.898
# qqp: 0.909
# mnli: 0.844
# qnli: 0.911
# rte: 0.708
# wnli: 0.563

ft_num_epochs = {
    "cola": 6,  # 8.5k, small dataset
    "sst2": 3,  # 67k, medium dataset
    "mrpc": 5,  # 3.7k, small dataset
    "stsb": 5,  # 5.7k, small dataset
    "qqp": 3,  # 364k, large dataset
    "mnli": 3,  # 393k, large dataset
    "qnli": 3,  # 105k, medium dataset
    "rte": 6,  # 2.5k, small dataset
    "wnli": 2,  # 0.6k, very small dataset
}
ft_lr = {
    "cola": 3e-5,
    "sst2": 3.5e-5,
    "mrpc": 3e-5,
    "stsb": 3e-5,
    "qqp": 5e-5,
    "mnli": 5e-5,
    "qnli": 5e-5,
    "rte": 2e-5,
    "wnli": 1e-5,
}
ft_weight_decay = {
    "cola": 0.01,
    "sst2": 0.01,
    "mrpc": 0.01,
    "stsb": 0.01,
    "qqp": 0.01,
    "mnli": 0.01,
    "qnli": 0.01,
    "rte": 0.01,
    "wnli": 0.01,
}
ft_batch_size = {
    "cola": 32,
    "sst2": 64,
    "mrpc": 16,
    "stsb": 16,
    "qqp": 256,
    "mnli": 256,
    "qnli": 128,
    "rte": 8,
    "wnli": 4,
}
# allow large batch size since bert base model parameters are frozen
ft_eval_steps = {
    "cola": 100,
    "sst2": 100,
    "mrpc": 50,
    "stsb": 100,
    "qqp": 200,
    "mnli": 200,
    "qnli": 100,
    "rte": 100,
    "wnli": 50,
}
ft_maxlen = 100   # according to bert-pretrained model
ft_num_cpus = 6

if imputation_mode == 'wordvec':
    glove_input_file = "glove.6B.50d.txt"  # Path to the GloVe file
    # Load GloVe directly with no_header=True
    word_vectors = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
#word2vec_output_file = "glove.6B.50d.word2vec.txt"  # Output path for Word2Vec format
# Load GloVe in Word2Vec format
#word_vectors = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# =============================================================

def preprocess_function_word_level(task, examples, tokenizer, max_length, corruption="none", ratio=0.0, imputation="none", imputation_model=None, imputation_noise=0.0):
    """
    Preprocess function to perturb input sentences at the word level by replacing words with [UNK] or random words.
    
    Args:
        examples (dict): Input examples.
        tokenizer (BertTokenizer): Tokenizer instance.
        max_length (int): Maximum sequence length.
        mode (str): Mode of modification. Options are 'miss' (replace with [UNK]) or 'noise' (replace with random words).
        ratio (float): Proportion of words to replace.

    Returns:
        dict: Tokenized input with perturbed sentences.
    """
    # Prepare random whole words if mode is 'noise'
    if corruption == 'noise' or imputation == 'insert_noise':
        # Filter for whole words in the vocabulary (no "##")
        whole_words = [word for word in tokenizer.vocab.keys() if not word.startswith("##")]

    # Helper function: corrupt sentence
    def perturb_sentence(sentence):
        """
        Perturb a single sentence by replacing words.
        """
        words = sentence.split()
        for i in range(len(words)):
            if random.random() < ratio:
                original_word = words[i]
                if corruption == 'miss':
                    words[i] = tokenizer.unk_token  # Replace with [UNK]
                    if imputation == 'bert':
                        words[i] = tokenizer.mask_token  # For BERT imputation, use [MASK]; otherwise use [UNK], later for imputation, BERT model will predict [UNK]!!
                    # Impute with noise if imputation mode is 'insert_noise'
                    if imputation == 'insert_noise':
                        if random.random() < imputation_noise:
                            words[i] = random.choice(whole_words)  # Replace with a random whole word
                        else:
                            words[i] = original_word  # Keep the original word
                elif corruption == 'noise':
                    words[i] = random.choice(whole_words)  # Replace with a random whole word
        return " ".join(words)

    # Helper function: impute sentence
    def impute_sentence(sentence):
        """
        Impute missing words (tokenizer.unk_token) in a sentence using word embeddings.
        
        Args:
            sentence (str): Input sentence with missing words represented by '[UNK]'.

        Returns:
            str: Sentence with tokenizer.unk_token replaced by imputed words.
        """
        words = sentence.split()
        imputed_words = []
        window_size = 2  # Number of words to consider on either side of [UNK]
        
        if imputation == 'wordvec':
            for i, word in enumerate(words):
                if word == tokenizer.unk_token:
                    # Collect context words using a sliding window
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(words), i + window_size + 1)
                    context = [
                        words[j]
                        for j in range(start_idx, end_idx)
                        if j != i and words[j] in word_vectors
                    ]
                    
                    if context:
                        # Compute the context vector as the mean of embeddings
                        context_vector = sum(word_vectors[c] for c in context) / len(context)
                        
                        # Find the most similar word in the vocabulary to the context vector
                        most_similar_word = word_vectors.similar_by_vector(context_vector, topn=1)[0][0]
                        # The method similar_by_vector returns a list of tuples where each tuple represents a word and 
                        # its similarity score. The structure of the output is:
                        # [
                        #    ('most_similar_word', similarity_score),
                        #    ('second_most_similar_word', similarity_score),
                        #    ...
                        #]
                        # hence "[0][0]" to get most_similar_word then the word itself
                        imputed_words.append(most_similar_word)
                    else:
                        # If no context is available, fall back to a placeholder
                        imputed_words.append(tokenizer.unk_token)
                else:
                    imputed_words.append(word)
            return " ".join(imputed_words)
        elif imputation == 'bert':
            # Tokenize the sentence
            inputs = tokenizer(sentence, return_tensors="pt")
            token_ids = inputs["input_ids"]

            # Predict using BERT
            with torch.no_grad():
                outputs = imputation_model(**inputs)
                predictions = outputs.logits

            # Replace each '[unk]' with the top predicted token
            for i, token_id in enumerate(token_ids[0]):
                if token_id == tokenizer.convert_tokens_to_ids('[CLS]') or token_id == tokenizer.convert_tokens_to_ids('[SEP]'):
                    continue
                
                if token_id == tokenizer.convert_tokens_to_ids('[MASK]'):
                    # Get the predicted token (top 1 prediction)
                    predicted_token_id = predictions[0, i].argmax(dim=-1).item()
                    predicted_word = tokenizer.decode([predicted_token_id]).strip()
                    imputed_words.append(predicted_word)
                else:
                    # Decode the original token
                    original_word = tokenizer.decode([token_id]).strip()
                    imputed_words.append(original_word)

            # Combine the imputed words into a sentence
            imputed_sentence = " ".join(imputed_words)

            # Post-process: Clean up formatting (remove extra spaces around punctuation)
            imputed_sentence = imputed_sentence.replace(" ##", "").replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            return imputed_sentence
        else:
            return Exception(f"Invalid imputation method: {imputation}")
        

    # Perturb, impute and tokenize the perturbed examples
    if task == 'qqp':
        # corrupt sentences
        if corruption != "none":
            examples["question1"] = [perturb_sentence(s) for s in examples["question1"]]
            examples["question2"] = [perturb_sentence(s) for s in examples["question2"]]
        # impute sentences
        if imputation == "wordvec" or imputation == "bert":
            examples["question1"] = [impute_sentence(s) for s in examples["question1"]]
            examples["question2"] = [impute_sentence(s) for s in examples["question2"]]
        # tokenize sentences
        tokenized = tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    elif task == 'mnli':
        # corrupt sentences
        if corruption != "none":
            examples["premise"] = [perturb_sentence(s) for s in examples["premise"]]
            examples["hypothesis"] = [perturb_sentence(s) for s in examples["hypothesis"]]
        # impute sentences
        if imputation == "wordvec" or imputation == "bert":
            examples["premise"] = [impute_sentence(s) for s in examples["premise"]]
            examples["hypothesis"] = [impute_sentence(s) for s in examples["hypothesis"]]
        # tokenize sentences
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    elif task == 'qnli':
        # corrupt sentences
        if corruption != "none":
            examples["question"] = [perturb_sentence(s) for s in examples["question"]]
            examples["sentence"] = [perturb_sentence(s) for s in examples["sentence"]]
        # impute sentences
        if imputation == "wordvec" or imputation == "bert":
            examples["question"] = [impute_sentence(s) for s in examples["question"]]
            examples["sentence"] = [impute_sentence(s) for s in examples["sentence"]]
        # tokenize sentences
        tokenized = tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    elif task in ["mrpc", "stsb", "rte", 'wnli']:
        # corrupt sentences
        if corruption != "none":
            examples["sentence1"] = [perturb_sentence(s) for s in examples["sentence1"]]
            examples["sentence2"] = [perturb_sentence(s) for s in examples["sentence2"]]
        # impute sentences
        if imputation == "wordvec" or imputation == "bert":
            examples["sentence1"] = [impute_sentence(s) for s in examples["sentence1"]]
            examples["sentence2"] = [impute_sentence(s) for s in examples["sentence2"]]
        # tokenize sentences
        tokenized = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    else:
        # corrupt sentences
        if corruption != "none":
            examples["sentence"] = [perturb_sentence(s) for s in examples["sentence"]]
        # impute sentences
        if imputation == "wordvec" or imputation == "bert":
            examples["sentence"] = [impute_sentence(s) for s in examples["sentence"]]
        # tokenize sentences
        tokenized = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    return tokenized


def preprocess_function_word_level_for_bert_imputation(task, examples, tokenizer, max_length, corruption="none", ratio=0.0, imputation="none", imputation_model=None, imputation_noise=0.0):
    """
    Preprocess function to perturb input sentences at the word level by replacing words with [UNK] or random words.
    To leverage GPU parallelism in your function, we need to vectorize operations wherever possible, and explicitly move tensor computations (like tokenization, corruption, and imputation) to the GPU. 
    
    Args:
        examples (dict): Input examples.
        tokenizer (BertTokenizer): Tokenizer instance.
        max_length (int): Maximum sequence length.
        mode (str): Mode of modification. Options are 'miss' (replace with [UNK]) or 'noise' (replace with random words).
        ratio (float): Proportion of words to replace.

    Returns:
        dict: Tokenized input with perturbed sentences.
    """
    # Helper function: corrupt sentence
    def perturb_sentences(sentences):
        """
        Corrupt sentences in batch.
        """
        tokenized = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(imputation_model.device)
        
        # Apply corruption based on ratio
        mask = torch.rand(input_ids.shape).to(imputation_model.device) < ratio
        pad_mask = (input_ids == tokenizer.pad_token_id)
        mask = mask * (~pad_mask)
        input_ids[mask] = tokenizer.mask_token_id
        
        # Filter the input_ids to keep only [MASK] and non-special tokens (excluding [CLS])
        filtered_input_ids = [
            [token_id for token_id in sentence if token_id == tokenizer.mask_token_id or token_id not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]]
            for sentence in input_ids
        ]
        # Now decode the filtered input_ids
        decoded = tokenizer.batch_decode(filtered_input_ids, skip_special_tokens=False)
        #return tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        return decoded

    # Helper function: impute sentence
    def impute_sentences(sentences):
        """
        Impute missing words (represented by [UNK]) in sentences using a model like BERT.
        """
        tokenized = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(imputation_model.device)
        input_ids = tokenized["input_ids"]
        mask_token_id = tokenizer.mask_token_id
        unk_token_id = tokenizer.unk_token_id
        
        # Replace [UNK] with [MASK] for imputation
        input_ids[input_ids == unk_token_id] = mask_token_id
        
        with torch.no_grad():
            outputs = imputation_model(**tokenized)
            predictions = outputs.logits.argmax(dim=-1)

        # Replace [MASK] tokens with predicted tokens
        input_ids[input_ids == mask_token_id] = predictions[input_ids == mask_token_id]
        ans = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return ans
        

    # Processing based on task type
    if task == "qqp":
        examples["question1"] = perturb_sentences(examples["question1"])
        examples["question2"] = perturb_sentences(examples["question2"])
        examples["question1"] = impute_sentences(examples["question1"])
        examples["question2"] = impute_sentences(examples["question2"])
        tokenized = tokenizer(
            examples["question1"], examples["question2"], padding="max_length", truncation=True, max_length=max_length
        )
    elif task == "mnli":
        examples["premise"] = perturb_sentences(examples["premise"])
        examples["hypothesis"] = perturb_sentences(examples["hypothesis"])
        examples["premise"] = impute_sentences(examples["premise"])
        examples["hypothesis"] = impute_sentences(examples["hypothesis"])
        tokenized = tokenizer(
            examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=max_length
        )
    elif task == "qnli":
        examples["question"] = perturb_sentences(examples["question"])
        examples["sentence"] = perturb_sentences(examples["sentence"])
        examples["question"] = impute_sentences(examples["question"])
        examples["sentence"] = impute_sentences(examples["sentence"])
        tokenized = tokenizer(
            examples["question"], examples["sentence"], padding="max_length", truncation=True, max_length=max_length
        )
    elif task in ["mrpc", "stsb", "rte", 'wnli']:
        examples["sentence1"] = perturb_sentences(examples["sentence1"])
        examples["sentence2"] = perturb_sentences(examples["sentence2"])
        examples["sentence1"] = impute_sentences(examples["sentence1"])
        examples["sentence2"] = impute_sentences(examples["sentence2"])
        tokenized = tokenizer(
            examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=max_length
        )
    else:
        examples["sentence"] = perturb_sentences(examples["sentence"])
        examples["sentence"] = impute_sentences(examples["sentence"])
        tokenized = tokenizer(
            examples["sentence"], padding="max_length", truncation=True, max_length=max_length
        )

    return tokenized




#model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(base_model_path)
print("Available GPUs:", torch.cuda.device_count())

# Dictionary to store results for each task
results = []

# Loop over each task
for task in glue_tasks:
    print('================')
    print(f'current task: {task}')
    
    # For test only
    #if task in ['cola', 'sst2', 'mrpc', 'stsb']:
    #    continue    
    
    # Define number of labels based on task
    if task == "stsb":
        num_labels = 1  # Regression task
    elif task == "mnli":
        num_labels = 3  # Three labels for MNLI
    else:
        num_labels = 2  # Binary classification for other tasks
        
    # Load the GLUE dataset
    print('loading data...')
    dataset = load_dataset(dataset_path, task)
    # Load subset for finetuning
    print(f"Dataset size before subset: {len(dataset['train'])}")
    print(f'Loading a subset of the dataset for fine-tuning with ratio: {subset_ratio}')
    # Take a subset of the training set based on the subset_ratio
    train_dataset = dataset['train']
    subset_size = int(len(train_dataset) * subset_ratio)
    print(f"Subset size: {subset_size}")
    # Shuffle the dataset and select the first `subset_size` examples
    train_dataset = train_dataset.shuffle(seed=42).select(range(subset_size))
    # Replace the training dataset with the subset
    dataset['train'] = train_dataset
        
    # preprocess and encode dataset
    print('preprocess and encode dataset...')
    if imputation_mode == 'bert':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = BertForMaskedLM.from_pretrained(base_model_path).to(device)
    else:
        base_model = None
    # Preprocess datasets
    encoded_dataset = {}
    preprocess_func = preprocess_function_word_level if imputation_mode != 'bert' else preprocess_function_word_level_for_bert_imputation
    for split in dataset:  # Iterate over the splits ('train', 'validation', 'test')
        encoded_dataset[split] = dataset[split].map(
            lambda x: preprocess_func(
                task, 
                x,
                tokenizer=tokenizer,
                max_length=ft_maxlen,
                corruption=corruption_mode if split == 'train' else 'none',  # Apply 'miss' or noise only to 'train'; 'miss' for missing words (use [UNK]), 'noise' for random words;
                ratio=corruption_ratio,
                imputation=imputation_mode if split == 'train' else 'none',
                imputation_model=base_model,
                imputation_noise=imputation_noise_level,
            ),
            batched=True,
            batch_size=128,
            num_proc=ft_num_cpus if imputation_mode != 'bert' else 1,
        )
    # clear memory
    del base_model
    torch.cuda.empty_cache()
    
    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{task}",
        num_train_epochs=ft_num_epochs[task],
        learning_rate=ft_lr[task],
        #warmup_steps=500,  # steps for warmup
        #warmup_ratio=0.1,  # % of total steps
        weight_decay=ft_weight_decay[task],
        #evaluation_strategy="epoch",
        evaluation_strategy="no",  # evaluate settings; no eval if running on HPC
        #eval_steps=ft_eval_steps[task], # evaluate settings; comment out if running on HPC
        save_strategy="no",
        save_total_limit=0,
        per_device_train_batch_size=ft_batch_size[task],
        per_device_eval_batch_size=ft_batch_size[task],
        #logging_dir=f"./logs/{task}",  # logging settings; comment out if running on HPC
        #logging_steps=1e7,  # logging settings; comment out if running on HPC
        #log_level="error",  # Suppress most logs except errors and evaluation  # logging settings; comment out if running on HPC
        #report_to=["tensorboard"],  # Log to TensorBoard  # logging settings; comment out if running on HPC
    )
    
    # Load metric using evaluate
    #metric = evaluate.load('glue', task)
    # Note: need to connect to huggingface here!
    metric = evaluate.load('./metrics/glue', task)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if task == "cola":
            # For CoLA, use Matthews correlation
            result = metric.compute(predictions=predictions.argmax(axis=1), references=labels)
            # MCC takes into account true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), 
            # providing a more balanced evaluation of the model’s performance.
            return {"matthews_correlation": result["matthews_correlation"]}
        
        elif task == "stsb":
            # For the regression task (STS-B), ensure the predictions are in the correct shape
            predictions = predictions[:, 0]  # Flatten predictions
            result = metric.compute(predictions=predictions, references=labels)
            
            # Handle STS-B correlations dynamically
            pearson = result.get("pearson")
            spearman = result.get("spearman")
            
            # Calculate the average correlation if both are available
            if pearson is not None and spearman is not None:
                average_correlation = (pearson + spearman) / 2
                return {"pearson": pearson, "spearman": spearman, "average_correlation": average_correlation}
            elif pearson is not None:
                return {"pearson": pearson}
            elif spearman is not None:
                return {"spearman": spearman}
            else:
                raise ValueError("Expected Pearson or Spearman correlation for STS-B, but none were found.")
            
        else:
            # For other tasks, use accuracy
            result = metric.compute(predictions=predictions.argmax(axis=1), references=labels)
            return {"accuracy": result["accuracy"]}


    # Initialize model
    # If you load a pre-trained BERT model without a classification head (e.g., from bert-base-uncased), 
    # the classification head is created from scratch and initialized randomly based on the number of labels (num_labels) you specify.
    print('creating model...')
    model = BertForSequenceClassification.from_pretrained(base_model_path, num_labels=num_labels)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # Freeze BERT layers if needed
    for param in model.bert.parameters():
        param.requires_grad = True

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"] if task != "mnli" else encoded_dataset["validation_matched"], # Use alidation_matched or validation_mismatched for MNLI
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    print('start training...')
    train_result = trainer.train()
    print('start evaluating...')
    eval_result = trainer.evaluate()

    # Log results
    if task == "cola":
        score = eval_result["eval_matthews_correlation"]
    elif task == "stsb":
        score = eval_result.get("eval_average_correlation", eval_result.get("eval_pearson", eval_result.get("eval_spearman")))
    else:
        score = eval_result["eval_accuracy"]
    results.append({"task": task, "score": score})
    print(f'task: {task}, score: {score:.4f}')
    

print('==============')
# Convert results to a DataFrame and print/report
results_df = pd.DataFrame(results)
print(results_df)

# Calculate the overall GLUE score as an average of all task scores, excluding WNLI if desired
glue_score = results_df["score"].mean() 
#glue_score = results_df[results_df["task"] != "wnli"]["score"].mean()  # Exclude WNLI if needed
results_df.loc["GLUE score"] = ["Average", glue_score]  # Add GLUE score row
print(f'GLUE score: {glue_score:.4f}')