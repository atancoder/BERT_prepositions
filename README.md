## Preposition BERT

Implement my own version of [BERT]([url](https://aclanthology.org/N19-1423.pdf)https://aclanthology.org/N19-1423.pdf) but with a twist. Since I don't have 
the resources to train a full fledged LLM, I wanted to just train the LLM to predict prepositions. 

## Usage
Create and activate the conda environment
```
mamba env create -f env.yml
conda activate bert
```

Download the HuggingFace Model Repo
```
git clone https://huggingface.co/atancoder/BERT_prepositions
```

Hyperparam Tuning
```
python main.py lr_grid_search
```

Train the model
```
python main.py train_model
```

See the model in action
```
>>> python main.py make_inference

Orig sentence: [CLS] i nod at him and go into my tent. [SEP]
Mask sentence: [CLS] i nod [MASK] him and go into my tent. [SEP]
Pred sentence: [CLS] i nod against him and go into my tent. [SEP]

Orig sentence: [CLS] someone had definitely been in our home. [SEP]
Mask sentence: [CLS] someone had definitely been [MASK] our home. [SEP]
Pred sentence: [CLS] someone had definitely been in our home. [SEP]

Orig sentence: [CLS] i put a lot of work into it. [SEP]
Mask sentence: [CLS] i put a lot [MASK] work gestures it. [SEP]
Pred sentence: [CLS] i put a lot on work to it. [SEP]

Orig sentence: [CLS] but you struck at the cs first during the lespin affair and... ` ` beth realised she'd done something terribly wrong. [SEP]
Mask sentence: [CLS] but you struck at the cs first hess the lespin affair and... ` ` beth realised she'd done something terribly wrong. [SEP]
Pred sentence: [CLS] but you struck at the cs first of the lespin affair and... ` ` beth realised she'd done something terribly wrong. [SEP]
```

## Training
I used the [books dataset]([url](https://huggingface.co/datasets/bookcorpus)), which is one of the datasets used in the paper. 

In this version of BERT, we target prepositions and either mask them out (P=50%), replace them with a random vocabulary word (P=30%), or leave as is (P=20%). 


## Architecture
The model represents words with embedding size = 768. The model has 4 transformer blocks, each with 4 heads. The model can support context lengths up to 1024 tokens but for training purposes, we only utilize 128 length sequences.

There's 2 versions of the model: one using pytorch's Transformers class (model_pytorch_transformers.py), and another where I build the transformer layer myself (model.py). For the latter, it's inspired by Andrej Karpathy's [basic transformer model](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

## Results

As of now, the model has learned to reliably predict prepositions for the prepositions we targeted. But it's not consistent at getting the correct preposition