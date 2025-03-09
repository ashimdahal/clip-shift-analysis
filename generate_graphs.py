import torch
import random 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from main import ImageTransformDataset, CLIPImageProcessor
from scipy.stats import gaussian_kde
from torchvision.transforms import ToPILImage
from PIL import Image

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



def l2_distance(u, v):
    """
    Compute the Euclidean (L2) distance between two vectors u and v.
    
    Args:
        u, v (list or np.array): The two vectors.
        
    Returns:
        float: The Euclidean distance.
    """
    u = np.array(u)
    v = np.array(v)
    return np.linalg.norm(u - v)

def plot_distance_bar_chart(embedding_pt_path, output_path="distance_bar_chart.png"):
    """
    Loads embeddings from the .pt file, computes the L2 distance between the original
    embedding and each augmented embedding for every sample, and plots a bar chart showing 
    the average L2 distance (with standard deviation error bars) for each augmentation.
    
    Args:
        embedding_pt_path (str or Path): Path to the .pt file containing the embeddings.
        output_path (str): File path to save the bar chart.
    """
    embeddings = torch.load(embedding_pt_path)
    
    # Expected structure: list of dicts, each with an "embeddings" dict that has keys like "original", "noise", "blur", etc.
    first_sample = embeddings[0]
    aug_keys = list(first_sample["embeddings"].keys())
    if "original" in aug_keys:
        aug_keys.remove("original")
    
    # Compute L2 distances for each augmentation per sample.
    distances_dict = {key: [] for key in aug_keys}
    for sample in embeddings:
        orig_emb = sample["embeddings"]["original"]
        for key in aug_keys:
            aug_emb = sample["embeddings"][key]
            dist = l2_distance(orig_emb, aug_emb)
            distances_dict[key].append(dist)
    
    # Calculate means and standard deviations.
    means = [np.mean(distances_dict[key]) for key in aug_keys]
    stds  = [np.std(distances_dict[key]) for key in aug_keys]
    
    # Create a polished bar chart.
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("pastel", len(aug_keys))
    bars = plt.bar(aug_keys, means, yerr=stds, capsize=8, color=colors, edgecolor='black')
    
    plt.xlabel("Augmentation Type", fontsize=12)
    plt.ylabel("Average L2 Distance", fontsize=12)
    plt.title("Average L2 Distance (Original vs. Augmented)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate each bar with the mean value.
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(means), 
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotate x-axis labels to avoid overlap.
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Distance bar chart saved to {output_path}")

def plot_distance_kde(embedding_pt_path, output_path="distance_kde.png"):
    """
    Loads embeddings from the .pt file, computes the L2 distance between the original
    embedding and each augmented embedding for every sample, and plots a kernel density 
    estimation (KDE) for the distance distribution of each augmentation. Each legend 
    label includes the augmentation’s mean and standard deviation.
    
    Additionally, the x-axis is set to start at -2, and the legend is placed to the left
    so that it doesn't overlap the plot.
    
    Args:
        embedding_pt_path (str or Path): Path to the .pt file containing the embeddings.
        output_path (str): File path to save the KDE plot.
    """
    embeddings = torch.load(embedding_pt_path)
    
    # Expected structure: list of dicts, each with an "embeddings" dict that has keys like "original", "noise", "blur", etc.
    first_sample = embeddings[0]
    aug_keys = list(first_sample["embeddings"].keys())
    if "original" in aug_keys:
        aug_keys.remove("original")
    
    # Compute L2 distances for each augmentation.
    distances_dict = {key: [] for key in aug_keys}
    for sample in embeddings:
        orig_emb = sample["embeddings"]["original"]
        for key in aug_keys:
            aug_emb = sample["embeddings"][key]
            dist = l2_distance(orig_emb, aug_emb)
            distances_dict[key].append(dist)
    
    # Plot a KDE for each augmentation with mean and std in the legend.
    plt.figure(figsize=(10, 6))
    for key in aug_keys:
        mean_val = np.mean(distances_dict[key])
        std_val = np.std(distances_dict[key])
        label = f"{key.replace('_',' ')}"
        
        # Compute KDE using scipy.stats.gaussian_kde
        data = np.array(distances_dict[key])
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 100)
        y_vals = kde(x_vals)
        
        # Plot KDE manually
        plt.plot(x_vals, y_vals, label=label, alpha=0.7)
        plt.fill_between(x_vals, y_vals, alpha=0.3)
        
        # Find peak and place augmentation label
        peak_x = x_vals[np.argmax(y_vals)]
        peak_y = max(y_vals)
        plt.text(peak_x, peak_y + 0.01, key.replace("_", " "), fontsize=8, ha="center")
    
    plt.xlabel("L2 Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("KDE of L2 Distance (Original vs. Augmented)", fontsize=14, fontweight='bold')
    
    # Set x-axis to start at -2.
    plt.xlim(-1, 18)
    
    # Place the legend outside the plot on the left.
    plt.legend(title="Augmentation Type", fontsize=10, title_fontsize=12, loc='upper right')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Distance KDE plot saved to {output_path}")


def get_attention_map(clip_processor, image):
    """
    Extracts the attention map from CLIP's last self-attention layer.

    Args:
        clip_processor (CLIPImageProcessor): CLIP processor object.
        image (PIL.Image): Input image.

    Returns:
        np.array: Attention map (resized to image size).
    """
    model = clip_processor.model
    device = clip_processor.device
    
    with torch.no_grad():
        # Process the image
        inputs = clip_processor.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get vision model
        vision_model = model.vision_model
        
        # We need to modify the forward call to return attentions
        # This is specific to the ViT-based CLIP model
        outputs = vision_model(
            pixel_values=inputs['pixel_values'],
            output_attentions=True
        )
        
        # Get the last layer attention
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("Model did not return attention maps. Check model configuration.")
        
        # Shape: [batch_size, num_heads, sequence_length, sequence_length]
        # We average over the heads dimension
        attention = outputs.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()
        
        # Get CLS token attention (first token's attention to all other tokens)
        cls_attention = attention[0, 1:]  # Skip the CLS token's attention to itself
        
        # Calculate grid size for the image patches
        patch_size = vision_model.config.patch_size
        img_size = vision_model.config.image_size
        grid_size = img_size // patch_size
        
        # Reshape to 2D grid (excluding CLS token)
        attn_map = cls_attention.reshape(grid_size, grid_size)
        
        # Resize to match original image size
        attn_map = cv2.resize(attn_map, (image.width, image.height))
        
        # Normalize for visualization
        if attn_map.max() > attn_map.min():  # Avoid division by zero
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        
        return attn_map


def overlay_attention(image, attention_map, alpha=0.5):
    """
    Overlays the attention heatmap on the original image.

    Args:
        image (PIL.Image): Original image.
        attention_map (np.array): Attention heatmap.
        alpha (float): Transparency of the overlay.

    Returns:
        PIL.Image: Image with overlay.
    """
    # Convert PIL to numpy
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Convert to uint8 for colormap application
    attention_uint8 = np.uint8(255 * attention_map)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Normalize heatmap
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    # Convert back to PIL
    return Image.fromarray(np.uint8(overlay * 255))


def plot_attention_maps(image_dir, clip_processor, dataset, output_path="attention_maps.png", num_samples=3):
    """
    Generates attention maps for original and augmented images and plots them.

    Args:
        image_dir (str): Directory containing images.
        clip_processor (CLIPImageProcessor): Initialized CLIP processor.
        dataset (ImageTransformDataset): Dataset object for augmentation.
        output_path (str): Path to save the final visualization.
        num_samples (int): Number of sample images to process.
    """
    # Get a list of available images
    image_paths = dataset.image_paths[:num_samples]  # Limit to specified number of samples
    
    num_images = len(image_paths)
    # Get augmentation keys (excluding image_path and original)
    sample = dataset[0]
    aug_keys = [key for key in sample.keys() if key not in ["image_path", "original"]]
    num_augments = len(aug_keys)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, num_augments + 2, figsize=(num_augments * 3 + 6, num_images * 3))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)  # Handle case with single image
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{num_images}: {image_path}")
        
        # Get index of this image in the dataset
        idx = dataset.image_paths.index(image_path)
        
        # Get transformed images
        sample = dataset[idx]
        orig_image = sample["original"]
        
        # Compute attention map for original image
        try:
            orig_attn = get_attention_map(clip_processor, orig_image)
            orig_overlay = overlay_attention(orig_image, orig_attn)
            
            # Get original embedding
            orig_embedding = clip_processor.get_embeddings([orig_image]).numpy()
            
            # Plot original image and its attention
            axes[i, 0].imshow(orig_image)
            axes[i, 0].set_title("Original", fontsize=12)
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(orig_overlay)
            axes[i, 1].set_title("Attention Map", fontsize=12)
            axes[i, 1].axis("off")
            
            # Process each augmentation
            for j, key in enumerate(aug_keys):
                aug_image = sample[key]
                
                try:
                    # Get attention map
                    aug_attn = get_attention_map(clip_processor, aug_image)
                    aug_overlay = overlay_attention(aug_image, aug_attn)
                    
                    # Compute similarity score
                    aug_embedding = clip_processor.get_embeddings([aug_image]).numpy()
                    cos_sim = np.dot(orig_embedding.flatten(), aug_embedding.flatten()) / (
                        np.linalg.norm(orig_embedding) * np.linalg.norm(aug_embedding)
                    )
                    
                    # Plot augmented image with attention overlay
                    axes[i, j + 2].imshow(aug_overlay)
                    axes[i, j + 2].set_title(f"{key}\nSim: {cos_sim:.2f}", fontsize=10)
                    axes[i, j + 2].axis("off")
                    
                except Exception as e:
                    print(f"Error processing augmentation {key} for image {i}: {e}")
                    axes[i, j + 2].text(0.5, 0.5, f"Error: {str(e)[:20]}...", 
                                      horizontalalignment='center', verticalalignment='center')
                    axes[i, j + 2].axis("off")
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error processing image: {str(e)[:50]}...", 
                          horizontalalignment='center', verticalalignment='center')
            axes[i, 0].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention visualization to {output_path}")
    plt.close()

def plot_attention_grids(image_dir, clip_processor, dataset, output_dir="attention_grids", 
                        num_graphs=5, num_samples_per_graph=7, fig_width=18, 
                        row_height=2.5, dpi=150):
    """
    Generates grid visualizations showing original images, augmentations, and attention maps.
    
    Args:
        image_dir (str): Directory containing images
        clip_processor (CLIPImageProcessor): Initialized CLIP processor
        dataset (ImageTransformDataset): Dataset object with augmentations
        output_dir (str): Output directory for saving grids
        num_graphs (int): Number of grid figures to generate
        num_samples_per_graph (int): Number of samples per grid
        fig_width (int): Width of each output figure in inches
        row_height (int): Height per row in inches
        dpi (int): Output image quality
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all available image paths
    all_image_paths = dataset.image_paths.copy()
    total_images = len(all_image_paths)
    
    # Calculate how many samples we can actually process
    max_possible_graphs = total_images // num_samples_per_graph
    actual_graphs = min(num_graphs, max_possible_graphs)
    
    if actual_graphs < num_graphs:
        print(f"Warning: Only enough data for {actual_graphs} graphs "
              f"(requested {num_graphs})")
    
    # Get augmentation keys (should be 9)
    sample = dataset[0]
    aug_keys = [key for key in sample.keys() if key not in ["image_path", "original"]]
    num_augments = len(aug_keys)
    
    if num_augments != 9:
        print(f"Warning: Found {num_augments} augmentations, expected 9. "
              "Layout may be affected.")
    
    # Main processing loop for each graph
    for graph_idx in range(actual_graphs):
        # Select images for this graph
        start_idx = graph_idx * num_samples_per_graph
        end_idx = start_idx + num_samples_per_graph
        graph_image_paths = all_image_paths[start_idx:end_idx]
        
        print(f"\nProcessing graph {graph_idx+1}/{actual_graphs} "
              f"({len(graph_image_paths)} samples)")
        
        # Create figure with appropriate dimensions
        fig_rows = 2 * len(graph_image_paths)  # 2 rows per sample
        fig_cols = 10  # Original + 9 augmentations
        
        fig, axes = plt.subplots(fig_rows, fig_cols, 
                                figsize=(fig_width, row_height * fig_rows),
                                dpi=dpi)
        
        # Hide initial axes
        for ax in axes.flatten():
            ax.axis('off')
        
        # Process each sample in current graph
        for sample_idx, image_path in enumerate(graph_image_paths):
            print(f"  Processing sample {sample_idx+1}: {Path(image_path).name}")
            row_base = sample_idx * 2  # Base row index for this sample
            
            try:
                # Get dataset index and load data
                idx = dataset.image_paths.index(image_path)
                sample_data = dataset[idx]
                orig_image = sample_data["original"]
                
                # Process original image
                orig_attn = get_attention_map(clip_processor, orig_image)
                orig_overlay = overlay_attention(orig_image, orig_attn)
                orig_embedding = clip_processor.get_embeddings([orig_image]).numpy()
                
                # Plot original image and attention
                axes[row_base, 0].imshow(orig_image)
                axes[row_base, 0].set_title("Original", fontsize=12)
                axes[row_base, 0].axis('off')
                
                axes[row_base+1, 0].imshow(orig_overlay)
                axes[row_base+1, 0].axis('off')
                
                # Process augmentations
                for aug_idx, aug_key in enumerate(aug_keys):
                    col = aug_idx + 1  # Columns 1-9
                    try:
                        aug_image = sample_data[aug_key]
                        
                        # Get attention and embedding
                        aug_attn = get_attention_map(clip_processor, aug_image)
                        aug_overlay = overlay_attention(aug_image, aug_attn)
                        aug_embedding = clip_processor.get_embeddings([aug_image]).numpy()
                        
                        # Plot augmented image
                        axes[row_base, col].imshow(aug_image)
                        title = f"{aug_key.replace('_',' ')}"
                        axes[row_base, col].set_title(title, fontsize=12)
                        axes[row_base, col].axis('off')
                        
                        # Plot attention overlay
                        axes[row_base+1, col].imshow(aug_overlay)
                        axes[row_base+1, col].axis('off')
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)[:15]}..."
                        axes[row_base, col].text(0.5, 0.5, error_msg,
                                                ha='center', va='center', fontsize=5)
                        axes[row_base, col].axis('off')
                        axes[row_base+1, col].axis('off')
                        
            except Exception as e:
                error_msg = f"Sample error: {str(e)[:20]}..."
                for col in range(fig_cols):
                    axes[row_base, col].text(0.5, 0.5, error_msg,
                                            ha='center', va='center', fontsize=6)
                    axes[row_base, col].axis('off')
                    axes[row_base+1, col].axis('off')
        
        # Finalize and save figure
        # plt.subplots_adjust(wspace=0.05, hspace=0.15)
        plt.subplots_adjust(
            wspace=0,  
            hspace=0.1,   # Reduce vertical spacing
            left=0,
            right=1,
            top=0.95,
            bottom=0
        )
        fig.tight_layout(pad=0, w_pad=0, h_pad=0.1)  # Squeeze everything
        fig.savefig(output_path / f"attention_grid_full_page_{graph_idx+1}.png", 
                   bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"Saved graph {graph_idx+1} to {output_path}/attention_grid_{graph_idx+1}.png")

if __name__ == "__main__":
    # Directory containing a single image (make sure this directory contains input_image.jpg)
    single_image_dir = "./input_dir/"  
    # plot_image_augmentations(single_image_dir, image_size=(224,224), output_path="visualization.png")

    embedding_file = Path("./clip_output/clip_embeddings_incremental.pt")

    # plot_cosine_similarity_distribution(embedding_file, output_path="cosine_similarity.png")
    # plot_cosine_similarity_heatmap(embedding_file, output_path="cosine_similarity_heatmap.png", sample_limit=50)

    # plot_distance_bar_chart(embedding_file, output_path="distance_bar_chart.png")
    # plot_distance_kde(embedding_file, output_path="distance_kde.png")
    images_dir = "./dataset/"  # Directory with a few sample images
    output_dir = Path("./visualization_output/")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize CLIP model
    clip_processor = CLIPImageProcessor(model_name="openai/clip-vit-base-patch32")
    
    # Create dataset with transformations
    dataset = ImageTransformDataset(
        image_dir=images_dir,
        image_size=(224, 224)
    )
    
    # Generate and save attention maps
    # plot_attention_maps(
    #     image_dir=images_dir,
    #     clip_processor=clip_processor,
    #     dataset=dataset,
    #     output_path=str(output_dir / "attention_maps_5.png"),
    #     num_samples=5  # Process 3 sample images
    # )

    plot_attention_grids(
        image_dir=images_dir,
        clip_processor=clip_processor,
        dataset=dataset,
        output_dir=output_dir,
        num_graphs=5,
        num_samples_per_graph=7,
        fig_width=23,
        row_height=2
    )
