import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer, RobertaModel
import argparse
import math

def compute_surprisal(sentence, target_token, model, tokenizer):
    """
    :param sentence: a sample sentence containing a verb of interest
    :param target_token: token at which surprisal is measured
    :param model: LLM of choice
    :param tokenizer: tokenizer
    :return: surprisal at the target token using pre-trained BERT
    """

    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    tokenized_sentence = tokenizer.tokenize(sentence)
    print(tokenized_sentence)
    target_index = tokenized_sentence.index(target_token)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    target_token_id = tokenizer.convert_tokens_to_ids(target_token)
    target_token_probability = probabilities[0, target_index, target_token_id].item()

    surprisal = -math.log2(target_token_probability)

    return surprisal


from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch


def compute_surprisal_roberta(sentence, target_token, model, tokenizer):
    """
    Computes the surprisal of a target token in a sentence using RoBERTa.
    """
    # Tokenize the sentence
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # Tokenize the target token to ensure matching
    tokenized_target = tokenizer.tokenize(target_token)  # Matches RoBERTa tokenization format
    target_length = len(tokenized_target)

    # Convert input IDs to tokens for inspection
    tokenized_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Search for the tokenized target in the tokenized sentence
    target_index = None
    for i in range(len(tokenized_sentence) - target_length + 1):
        if tokenized_sentence[i : i + target_length] == tokenized_target:
            target_index = i
            break

    # if target_index is None:
    #     raise ValueError(
    #         f"'{target_token}' is not in the tokenized sentence: {tokenized_sentence}\n"
    #         f"Tokenized target: {tokenized_target}"
    #     )

    # Mask the target token in the sentence
    masked_input_ids = input_ids.clone()
    for i in range(target_length):
        masked_input_ids[0, target_index + i] = tokenizer.mask_token_id

    # Get model outputs for the masked sentence
    with torch.no_grad():
        outputs = model(masked_input_ids)
        logits = outputs.logits

    # Compute probability of the target token(s)
    log_probs = 0.0
    for i, token_id in enumerate(input_ids[0, target_index : target_index + target_length]):
        token_logits = logits[0, target_index + i]
        token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
        log_probs += token_log_prob.item()

    # Surprisal is the negative log probability
    surprisal = -log_probs
    return surprisal


def main():
    print("Script started")
    parser = argparse.ArgumentParser(description="Compute surprisal for sentences using BERT.")
    parser.add_argument("--sentences", type=str, nargs="+", help="Sentences")
    parser.add_argument("--target_tokens", type=str, nargs="+", help="Target tokens")
    args = parser.parse_args()

    if len(args.sentences) != len(args.target_tokens):
        raise ValueError(f"Number of sentences and tokens do not match. {[s for s in args.sentences]} sentences and {len(args.target_tokens)} tokens")

    #model_name = #"bert-large-uncased" #"bert-base-uncased"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large") #AutoTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained("roberta-large") #AutoModelForMaskedLM.from_pretrained(model_name)

    for sentence, target_token in zip(args.sentences, args.target_tokens):
        surprisal = compute_surprisal_roberta(sentence, target_token, model, tokenizer)

        print(f"Sentence {sentence}")
        print(f"Target Token: {target_token}")
        print(f"Surprisal: {surprisal:.4f}\n")

if __name__ == "__main__":
    main()


