# DA6401-Assignment-3
# Name: M S SAI TEJA
# Roll No.: ME21B171
# Link to wandb link: [Wandb_Report](https://wandb.ai/teja_sai-indian-institute-of-technology-madras/transliteration-seq2seq/reports/DA6401-Assignment-3--VmlldzoxMjg2MDEyNg?accessToken=nd2vybf27l77p6bklyx8kav72bgk2ajsovdwbrxilbvn4ozbgh9erudqps5fk9tq)
# Tamil Transliteration with Sequence-to-Sequence Models

This repository contains the implementation of sequence-to-sequence models for Tamil transliteration as part of the DA6401 Deep Learning Assignment 3. The project explores both vanilla sequence-to-sequence models and attention-based models for transliterating romanized Tamil text to native Tamil script.

## Repository Structure

```
DA6401-Assignment-3/
├── configs/                  # Configuration files
├── data/                     # Data loading and preprocessing
├── models/                   # Model definitions
├── predictions/              # Model predictions
│   ├── vanilla/              # Vanilla model predictions
│   └── attention/            # Attention model predictions
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   ├── models/               # Model implementation
│   ├── training/             # Training logic
│   └── visualization/        # Visualization utilities
├── visualization/            # Visualization outputs
├── main.py                   # Main script
├── visualization.py          # Visualization script
├── connectivity_visualization.py  # Connectivity visualization script
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── Transliteration_Assignment.ipynb
```
This is the high-level structure of the repo. There can be minute differences, but that won't cause any issue.
##  Overview of the models 

This repo implements and compares two sequence-to-sequence architectures for Tamil transliteration:

1. **Vanilla Seq2Seq**: A basic encoder-decoder architecture
2. **Attention-based Seq2Seq**: An enhanced model with an attention mechanism

We perform hyperparameter tuning for both models, analyze their performance, and visualize the attention mechanism to understand how the model learns to map between romanized and native Tamil script.

## Dataset

We have used the standard train, dev, test set from the folder dakshina_dataset_v1.0/ta/lexicons.

## Model Architecture

### Vanilla Seq2Seq

- Encoder: RNN/LSTM/GRU with configurable layers and hidden size
- Decoder: RNN/LSTM/GRU with configurable layers and hidden size
- Teacher forcing during training


### Attention-based Seq2Seq

- Encoder: Similar to vanilla model
- Decoder: Enhanced with an attention mechanism that allows it to focus on different parts of the input sequence
- Attention weights visualization for model interpretability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/saiteja-ms/DA6401-Assignment-3.git
cd DA6401-Assignment-3
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the Dakshina dataset:
```bash
# The dataset is already be organized in the following structure:
# ./ta/lexicons/ta.translit.sampled.train.tsv
# ./ta/lexicons/ta.translit.sampled.dev.tsv
# ./ta/lexicons/ta.translit.sampled.test.tsv

## Usage

### Training

To train a model with default parameters:

```bash
python main.py --mode train --model_type vanilla --language ta --data_dir "."
```

For attention-based model:

```bash
python main.py --mode train --model_type attention --language ta --data_dir "."
```

To train with a specific configuration:

```bash
python main.py --mode train --model_type vanilla --language ta --data_dir "." --config configs/vanilla_config.json
```


### Hyperparameter Tuning

To run a hyperparameter sweep:

```bash
python main.py --mode sweep --model_type vanilla --language ta --data_dir "." --sweep_count 50
```


### Evaluation

To evaluate a trained model:

```bash
python main.py --mode analyze --vanilla_run_id YOUR_VANILLA_RUN_ID --attention_run_id YOUR_ATTENTION_RUN_ID --language ta --data_dir "."
```


### Visualization

To generate attention heatmaps:

```bash
python visualization.py --predictions_file predictions/attention/predictions-YOUR_RUN_ID.json --visualization_type heatmap
```

To generate connectivity visualizations:

```bash
python connectivity_visualization.py --predictions_file predictions/attention/predictions-attention_best_predictions.json
```

You can also run the "Transliteration_Assignment.ipynb" which also implements the training and hyperparameter sweeps for the vanilla and attention based models.

## Results

Our best vanilla model achieved an accuracy of 42.38% on the test set, while the attention-based model reached 58.15%, demonstrating the effectiveness of the attention mechanism for transliteration tasks.

Key findings:

- Attention mechanism significantly improves performance on longer sequences
- The model handles vowel markers more accurately with attention
- Visualization reveals clear patterns in how the model attends to different parts of the input



