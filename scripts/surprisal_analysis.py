import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
from transformers import BertTokenizer, BertForMaskedLM

# Load BERT-base-uncased
MODEL_NAME = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

# Define the allowed diagnostic types
VALID_DIAGNOSTICS = ["out_prefixation", "denial_of_action", "denial_of_result", "resultative", "object_omission"]

# Set a color palette for better differentiation
COLORS = COLORS = ['red', 'green', 'red', 'green', 'purple'] # manner in red, result in green, control in purple

def compute_surprisal(sentence, model, tokenizer):
    """
    Computes surprisal values for each token in the sentence.

    :param sentence: The input sentence
    :param model: Pre-trained BERT model
    :param tokenizer: Tokenizer for the model
    :return: List of (token, surprisal) tuples
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    tokenized_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])

    surprisals = []

    for i in range(1, len(tokenized_sentence) - 1):  # Exclude [CLS] and [SEP]
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id  # Mask the target word

        with torch.no_grad():
            outputs = model(masked_input_ids)
            logits = outputs.logits

        token_logits = logits[0, i]
        token_probs = torch.nn.functional.softmax(token_logits, dim=-1)
        target_token_id = input_ids[0, i].item()
        target_prob = token_probs[target_token_id].item()

        surprisal = -math.log2(target_prob)
        surprisals.append((tokenized_sentence[i], surprisal))

    return surprisals


def process_diagnostic(csv_file, diagnostic):
    """
    Reads sentences for a specific diagnostic, computes surprisals, generates a figure with 5 subplots,
    and saves the surprisal values to a CSV file.

    :param csv_file: Path to the CSV file containing verb examples
    :param diagnostic: The name of the diagnostic column to analyze
    """
    df = pd.read_csv(csv_file, index_col=0)
    verbs = df.index[:4]  # First 4 verbs
    control_verb = df.index[-1]  # Last row (assumed to be "thought")

    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

    all_surprisal_data = []
    all_surprisals = []

    for i, verb in enumerate(verbs):
        sentence = df.loc[verb, diagnostic]
        surprisal_values = compute_surprisal(sentence, model, tokenizer)
        tokens, surprisals = zip(*surprisal_values)
        print(tokens)

        axes[i].plot(range(len(tokens)), surprisals, label=f"{verb}", linestyle='-', marker='o', color=COLORS[i])
        axes[i].set_xticks(range(len(tokens)))
        axes[i].set_xticklabels(tokens, fontsize=8)  #rotation=45, ha="right"
        axes[i].set_ylabel("Surprisal (bits)")
        axes[i].grid(True)

        # Save surprisal values
        for token, surprisal in surprisal_values:
            all_surprisal_data.append([token, surprisal, verb])
            all_surprisals.append(surprisal) # for global max/min

    # Add control sentence ("thought") in the last subplot
    control_sentence = df.loc[control_verb, diagnostic]
    surprisal_values = compute_surprisal(control_sentence, model, tokenizer)
    tokens, surprisals = zip(*surprisal_values)

    axes[4].plot(range(len(tokens)), surprisals, linestyle='--', marker='s', color=COLORS[4])
    axes[4].set_ylabel("Surprisal (bits)")
    axes[4].set_xticks(range(len(tokens)))
    axes[4].set_xticklabels(tokens, fontsize=8) #rotation=45, ha="right"
    axes[4].grid(True)

    # Save control surprisal values
    for token, surprisal in surprisal_values:
        all_surprisal_data.append([token, surprisal, control_verb])

    # Set shared min and max value for y-axis
    global_min = min(all_surprisals)
    global_max = max(all_surprisals)
    for ax in axes:
        ax.set_ylim(global_min, global_max)

    # Save figure
    plt.tight_layout()
    figure_path = f"../results/figures/surprisal_{diagnostic}.png"
    plt.savefig(figure_path)
    plt.show()

    print(f"Figure saved to {figure_path}")

    # Save surprisal values to CSV
    surprisal_df = pd.DataFrame(all_surprisal_data, columns=["Token", "Surprisal", "Verb"])
    surprisal_csv_path = f"../results/surprisal/surprisal_values_{diagnostic}.csv"
    surprisal_df.to_csv(surprisal_csv_path, index=False)

    print(f"Surprisal values saved to {surprisal_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and visualize surprisal for a specific diagnostic.")
    parser.add_argument("diagnostic", type=str, choices=VALID_DIAGNOSTICS, help="The diagnostic type to analyze")
    args = parser.parse_args()

    csv_file = "../data/diagnostics/surprisal/surprisal_examples.csv"

    process_diagnostic(csv_file, args.diagnostic)
