import torch
import numpy as np
import difflib
import pandas as pd
import pickle
import argparse
from transformers import BertTokenizer, BertModel

# Load BERT model & tokenizer


def get_verb_representation(sentence, verb_lemma, model, tokenizer, layer_num):
    """
    Extracts BERT representation of a verb in a given sentence.
    Identifies the target token using lemma-based matching and subword handling.
    """
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    tokenized_sentence = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Step 1: Try to find an exact match for the lemma
    subword_indices = [i for i, token in enumerate(tokenized_sentence) if token.lstrip("##") == verb_lemma]

    # Step 2: Handle subword tokens if no exact match
    if not subword_indices:
        reconstructed = ""
        temp_indices = []

        for i, token in enumerate(tokenized_sentence):
            clean_token = token.lstrip("##")
            temp_indices.append(i)
            reconstructed += clean_token

            # Check if reconstructed token matches lemma
            if reconstructed == verb_lemma:
                subword_indices = temp_indices
                break

    # Step 3: If still no match, select the token with the highest character overlap
    if not subword_indices:
        best_match_idx = max(
            range(len(tokenized_sentence)),
            key=lambda i: difflib.SequenceMatcher(None, verb_lemma, tokenized_sentence[i].lstrip("##")).ratio()
        )
        subword_indices = [best_match_idx]

    # Extract embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # Shape: (num_layers, batch_size, seq_len, hidden_dim)

    selected_layer = hidden_states[layer_num].squeeze(0) # Shape: (seq_len, hidden_dim)

    # Aggregate embeddings (average across subwords if needed)
    representation = selected_layer[subword_indices].mean(dim=0).numpy()

    return representation


def compute_average_embedding(verb_lemma, examples, model, tokenizer, layer_num, max_examples=20):
    """
    Computes the average embedding for a given verb across multiple examples.
    """
    embeddings = []

    for sentence in examples[:max_examples]:  # Limit to first `max_examples` sentences
        rep = get_verb_representation(sentence, verb_lemma, model, tokenizer, layer_num)
        embeddings.append(rep)

    # Compute average representation
    avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(model.config.hidden_size)

    return avg_embedding


def process_verbs(csv_file, output_file, model_name, layer_num):
    """
    Reads a CSV file with verb examples, extracts representations, and saves them.
    """

    df = pd.read_csv(csv_file, index_col=0, encoding='ISO-8859-1')  # Read CSV with verbs as index
    verb_embeddings = {}

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    for verb_lemma in df.index:
        examples = df.loc[verb_lemma].dropna().tolist()  # Get non-null example sentences
        avg_embedding = compute_average_embedding(verb_lemma, examples, model, tokenizer, layer_num)
        verb_embeddings[verb_lemma] = avg_embedding

    # Save embeddings as a .pkl file for later use
    with open(output_file, "wb") as f:
        pickle.dump(verb_embeddings, f)

    print(f"Saved embeddings to {output_file}")


def main(model_name, layer_num, examples_file):

    output_file = f"../data/embs/{model_name}_{layer_num}_verb_embeddings.pkl"

    process_verbs(examples_file, output_file, model_name, layer_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument("--layer_num", type=int, default=6,
                        help="Path to the embeddings file")
    parser.add_argument("--examples_file", type=str, default="../data/diagnostics/example_usage/manres_examples_complete.csv", help="Path to the file with example usages of verbs")

    args = parser.parse_args()

    main(args.model_name, args.layer_num, args.examples_file)

