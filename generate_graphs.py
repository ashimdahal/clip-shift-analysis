import torch
import random 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from main import ImageTransformDataset

def plot_image_augmentations(image_dir, image_size=(512, 512), output_path="augmentation_visualization.png"):
    """
    Loads a single image from image_dir, applies augmentations using ImageTransformDataset,
    and plots the original and augmented images in one row with attractive styling.

    Args:
        image_dir (str): Directory containing a single image (e.g., "input_image.jpg").
        image_size (tuple): Size to which images are resized (default is (512, 512)).
        output_path (str): File path to save the generated plot.
    """
    # Set a seaborn style for a polished look
    sns.set(style="whitegrid", context="talk")

    # Create dataset with the given image size.
    dataset = ImageTransformDataset(image_dir=image_dir, image_size=image_size)
    
    # Get the first (and only) sample
    sample = dataset[0]

    # Extract keys and order them so that "original" is always first.
    keys = [key for key in sample.keys() if key != "image_path"]
    if "original" in keys:
        keys.remove("original")
        keys = ["original"] + keys

    n = len(keys)
    
    # Create a figure with one row and n columns.
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3), constrained_layout=True)
    
    # If only one subplot, ensure axes is a list for consistency.
    if n == 1:
        axes = [axes]
        
    # Plot each augmentation image with its label.
    for ax, key in zip(axes, keys):
        ax.imshow(sample[key])
        ax.set_title(key.capitalize(), fontsize=12, fontweight="bold", color="#333333")
        ax.axis("off")
    
    # Add an overall title to the figure.
    fig.suptitle("Original Image and Augmentations", fontsize=16, fontweight="bold", color="#444444")
    
    # Save the figure and display it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to {output_path}")

def cosine_similarity(u, v):
    """
    Compute cosine similarity between two vectors u and v.
    
    Args:
        u (list or np.array): First vector.
        v (list or np.array): Second vector.
    
    Returns:
        float: Cosine similarity value.
    """
    u = np.array(u)
    v = np.array(v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)

def plot_cosine_similarity_distribution(embedding_pt_path, output_path="cosine_similarity_distribution.png"):
    """
    Loads saved embeddings from a .pt file, computes cosine similarities between 
    the original embedding and each augmented embedding, and plots the distribution 
    of cosine similarity scores for each augmentation type with mean and std in the legend.
    
    Args:
        embedding_pt_path (str or Path): Path to the .pt file containing the embeddings.
        output_path (str): File path to save the distribution plot.
    """
    embeddings = torch.load(embedding_pt_path)
    
    # Expected structure: list of dicts, each with keys "image_path" and "embeddings"
    # where "embeddings" is a dict with keys like "original", "noise", "blur", etc.
    sim_dict = {}
    
    # Get augmentation keys (excluding "original")
    first_sample = embeddings[0]
    aug_keys = list(first_sample["embeddings"].keys())
    if "original" in aug_keys:
        aug_keys.remove("original")
    
    for key in aug_keys:
        sim_dict[key] = []
    
    # Compute cosine similarity for each sample and each augmentation type.
    for sample in embeddings:
        orig_emb = sample["embeddings"]["original"]
        for key in aug_keys:
            aug_emb = sample["embeddings"][key]
            sim = cosine_similarity(orig_emb, aug_emb)
            sim_dict[key].append(sim)
    
    # Plot a histogram for each augmentation type, including mean and std in the legend.
    plt.figure(figsize=(10, 6))
    for key, sims in sim_dict.items():
        mean_val = np.mean(sims)
        std_val = np.std(sims)
        label = f"{key.replace('_',' ')} (mean={mean_val:.2f}, std={std_val:.2f})"
        sns.histplot(sims, kde=True, label=label, bins=50, alpha=0.6)
    
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Similarity (Original vs. Augmented)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Cosine similarity distribution plot saved to {output_path}")

def plot_cosine_similarity_heatmap(
    embedding_pt_path,
    output_path="cosine_similarity_heatmap.png",
    sample_limit=50
):
    """
    Loads saved embeddings, computes cosine similarities between the original embedding 
    and each augmented embedding for a subset of samples, and plots a heatmap where rows 
    correspond to sample indices and columns correspond to augmentation types.
    
    Args:
        embedding_pt_path (str or Path): Path to the .pt file containing the embeddings.
        output_path (str): File path to save the heatmap plot.
        sample_limit (int): Number of samples to include in the heatmap for clarity.
    """
    embeddings = torch.load(embedding_pt_path)
    
    # Get augmentation keys (excluding "original")
    first_sample = embeddings[0]
    aug_keys = list(first_sample["embeddings"].keys())
    if "original" in aug_keys:
        aug_keys.remove("original")
    
    # If too many samples, randomly sample a subset
    if len(embeddings) > sample_limit:
        embeddings = random.sample(embeddings, sample_limit)
    
    # Build a 2D array: rows = samples, columns = augmentation types
    heatmap_data = []
    for sample in embeddings:
        row = []
        orig_emb = sample["embeddings"]["original"]
        for key in aug_keys:
            aug_emb = sample["embeddings"][key]
            sim = cosine_similarity(orig_emb, aug_emb)
            row.append(sim)
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)
    
    # Plot heatmap with Seaborn
    plt.figure(figsize=(len(aug_keys) * 1.5, sample_limit * 0.2 + 3))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", 
                     cbar_kws={'label': 'Cosine Similarity'},
                     xticklabels=aug_keys, yticklabels=False)
    plt.xlabel("Augmentation Type")
    plt.ylabel("Sample Index")
    plt.title("Cosine Similarity Heatmap (Original vs. Augmented)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Cosine similarity heatmap saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Directory containing a single image (make sure this directory contains input_image.jpg)
    single_image_dir = "./input_dir/"  
    # plot_image_augmentations(single_image_dir, image_size=(224,224), output_path="visualization.png")

    embedding_file = Path("./clip_output/clip_embeddings_incremental.pt")

    plot_cosine_similarity_distribution(embedding_file, output_path="cosine_similarity.png")
    plot_cosine_similarity_heatmap(embedding_file, output_path="cosine_similarity_heatmap.png", sample_limit=50)
