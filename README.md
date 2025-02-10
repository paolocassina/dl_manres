## Testing linguistics diagnostics with BERT

## Description
A project to explore Manner/Result complementarity (or lack thereof) in verb representations of BERT.


## Usage

### Data

#### Diagnostics/example_usage
This folder contains two csv files. manres_examples.csv contains example usages of each verb extracted from the SemCor 3.0 corpus. Some verbs have hundreds of examples, others have only a few, or none at all. Since I wanted to have a minimum of 20 examples per verb for probing, I manually selected additional examples for each verb that had fewer than 20 example sentences. I selected the additional examples from two corpora, both accessed via No Sketch Engine: Desert Island Disk, and UK Wac Complete. manres_examples_complete.csv contains the original examples from the SemCor 3.0 corpus, plus these additional examples mentioned above. manres_examples.complete.csv was the file used to obtain average BERT embeddings for our verbs

#### Diagnostics/surprisal
This folder contains a single file, surprisal_examples.csv, which contains sentences that work as diagnostics for Manner/Result complementarity, which were used for the surprisal analysis.

### Scripts


#### extract_representations.py
Extracts contextualized word embeddings from BERT for target verbs.
Supports different BERT models (e.g., bert-base, bert-large).
Can specify the BERT layer at which embeddings are extracted.
Saves extracted embeddings for further analysis.

Example usage:
```ruby
python extract_representations.py --model_name "bert-base-uncased" --layer_num 6
```
 

#### visualize_embeddings.py
Visualizes verb representations using t-SNE, a dimensionality reduction technique.
Helps interpret how verbs cluster in high dimensional space.

Example usage:
```ruby
python visualized_embeddings.py --embedding_file "../data/embs/bert-base-uncased_6_verb_embeddings.pkl"
```

#### probe_analysis.py
Trains a logistic regression classifier to predict manner/result labels from BERT embeddings.
Uses Stratified K-Fold cross-validation for evaluation.
Computes accuracy, F1-score, and AUC-ROC to assess model performance.

Example usage:
```ruby
python probe_analysis.py --embedding_file "../data/embs/bert-base-uncased_6_verb_embeddings.pkl" --pca False
```

#### surprisal_analysis.py
For each diagnostics sentence, it computes surprisal scores at every token.
Generates plots showing word-by-word surprisal in each diagnostics sentence.
Saves results as CSV files and visualizes surprisal across verbs.

Example usage:
```ruby
python surprisal_analysis.py --model_name "bert-large-uncased" --diagnostic "object_omission"
```


