import os.path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Load embeddings from the stored pickle file
def load_embeddings(embedding_file):
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


# Load the verb classes from the CSV file
def load_verb_classes(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    return dict(zip(df.index, df["class"]))  # Map verb lemma -> class


# Perform t-SNE on the embeddings
def tsne_reduce(embeddings, perplexity=10, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings


# Plot the embeddings
def plot_embeddings(embeddings_2d, labels, output_file):
    plt.figure(figsize=(10, 6))
    colors = {"manner": "red", "result": "green"}

    for (point, verb, label) in zip(embeddings_2d, labels.keys(), labels.values()):
        plt.scatter(point[0], point[1], color=colors[label],
                    label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(point[0] + 0.2, point[1], verb, fontsize=9)

    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    #plt.title("t-SNE: Manner/Result")
    plt.legend()
    plt.savefig(output_file)
    plt.show()


def main():
    embs_file = "bert-base-uncased_2_verb_embeddings.pkl"
    embeddings = load_embeddings(os.path.join("../data/embs/", embs_file))
    verb_classes = load_verb_classes("../data/verbs/manres_lemma.csv")

    # Ensure that we only use verbs present in both embeddings and class labels
    common_verbs = list(set(embeddings.keys()) & set(verb_classes.keys()))
    print(embeddings.keys())
    print(verb_classes.keys())
    print(f"Number of verbs in dataset: {len(common_verbs)}")
    embedding_matrix = np.array([embeddings[verb] for verb in common_verbs])
    labels = {verb: verb_classes[verb] for verb in common_verbs}

    # Reduce dimensionality and plot
    embeddings_2d = tsne_reduce(embedding_matrix)
    model_embs= embs_file.split("_")[0]
    layer_embs= embs_file.split("_")[1]

    plot_embeddings(embeddings_2d, labels, output_file=f"../results/figures/{model_embs}_{layer_embs}_manres_tsne.png")


if __name__ == "__main__":
    main()
