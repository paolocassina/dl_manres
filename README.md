# Testing linguistics diagnostics with BERT

## Description
A project to explore Manner/Result complementarity (or lack thereof) in verb representations of BERT.


## Usage

### Data

#### Diagnostics/example_usage
This folder contains two csv files. manres_examples.csv contains example usages of each verb extracted from the [SemCor](https://web.eecs.umich.edu/~mihalcea/downloads.html#semcor) 3.0 corpus. In this corpus, each verb is tagged with its WordNet sense, which allows to control for polysemy by extracting example usages of a verb only when  I found hundreds of examples for some verbs, while only a handful of examples for other verbs. Since I wanted to have a minimum of 20 examples per verb for probing, I manually selected additional examples for each verb that had fewer than 20 example sentences in SemCor. I selected the additional examples from two corpora, both accessed via [No Sketch Engine](https://bellatrix.sslmit.unibo.it/noske/public/#dashboard): Desert Island Discs corpus, and UK Wac Complete. manres_examples.csv contains the original examples from the SemCor 3.0 corpus. manres_examples_complete.csv contains the examples from SemCor 3.0 plus these additional examples from the other two corpora. manres_examples.complete.csv was the file used to obtain average BERT embeddings for our verbs.

#### Diagnostics/surprisal
This folder contains a single file, surprisal_examples.csv, which contains sentences that work as diagnostics for Manner/Result complementarity, which were used for the surprisal analysis.

#### Embs
Stores pkl files containing verb embeddings extracted through the extract_representation.py script. The name of the file specifies the model and the layer at which the embeddings were extracted.
For example, bert-base-uncased_8_verb_embeddings.pkl contains verb embeddings extracted from layer 8 of BERT base (12 layers), case insensitive.


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


