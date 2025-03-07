import torch
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from generate_graphs import get_attention_map, cosine_similarity
from main import ImageTransformDataset, CLIPImageProcessor

def plot_augmentation_dendrogram(clip_processor, dataset, output_path="dendrogram.png", num_samples=5, method='ward'):
    """
    Creates a dendrogram showing hierarchical clustering of augmentations based on 
    how they affect CLIP embeddings.
    
    Args:
        clip_processor (CLIPImageProcessor): Initialized CLIP processor
        dataset (ImageTransformDataset): Dataset object with augmentations
        output_path (str): Path to save the visualization
        num_samples (int): Number of sample images to use for averaging
        method (str): Linkage method for hierarchical clustering
    """
    # Get sample and augmentation keys
    sample = dataset[0]
    aug_keys = [key for key in sample.keys() if key not in ["image_path", "original"]]
    num_augments = len(aug_keys)
    
    # Store average embedding distances
    avg_distances = np.zeros((num_augments, num_augments))
    
    # Process multiple images and average the results
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        orig_image = sample["original"]
        
        # Get embeddings for all augmentations
        embeddings = {}
        embeddings["original"] = clip_processor.get_embeddings([orig_image]).numpy()
        
        for key in aug_keys:
            aug_image = sample[key]
            embeddings[key] = clip_processor.get_embeddings([aug_image]).numpy()
        
        # Compute distance matrix for this sample
        distance_matrix = np.zeros((num_augments, num_augments))
        for i, key1 in enumerate(aug_keys):
            for j, key2 in enumerate(aug_keys):
                if i == j:
                    continue
                # Use cosine distance (1 - cosine similarity)
                cos_sim = np.dot(embeddings[key1].flatten(), embeddings[key2].flatten()) / (
                    np.linalg.norm(embeddings[key1]) * np.linalg.norm(embeddings[key2])
                )
                distance_matrix[i, j] = 1 - cos_sim
        
        # Add to average
        avg_distances += distance_matrix
    
    # Calculate average
    if num_samples > 0:
        avg_distances /= min(num_samples, len(dataset))
    
    # Create a condensed distance matrix (required by linkage)
    condensed_dist = squareform(avg_distances)
    
    # Compute linkage matrix
    Z = linkage(condensed_dist, method=method)
    
    # Set up the plot with seaborn styling
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Calculate threshold for coloring
    max_d = 0.7 * max(Z[:, 2])
    
    # Plot dendrogram with improved labeling
    R = dendrogram(
        Z,
        labels=[key.replace('_', ' ').title() for key in aug_keys],
        orientation='top',
        leaf_rotation=45,  # 45 degrees instead of 90
        leaf_font_size=12,
        color_threshold=max_d
    )
    
    # Highlight clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    num_clusters = len(set(clusters))
    
    # Draw horizontal line at threshold
    plt.axhline(y=max_d, color='crimson', linestyle='--', alpha=0.8)
    plt.text(len(aug_keys)/2 +15, max_d + 0.02, f'Cluster Threshold: {max_d:.2f}', 
             va='bottom', ha='center', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             fontsize=12)
    
    # Get dendrogram structure for precise positioning
    icoord = np.array(R['icoord'])
    dcoord = np.array(R['dcoord'])
    
    # Add annotations for main clusters
    for i, cluster_id in enumerate(sorted(set(clusters))):
        indices = [j for j, x in enumerate(clusters) if x == cluster_id]
        if len(indices) > 1:
            # Find all horizontal bars in this cluster
            cluster_nodes = []
            for idx in indices:
                leaf_pos = R['leaves'].index(idx)
                # Find which horizontal bars contain this leaf
                for j, (x1, x2) in enumerate(zip(icoord[:,0], icoord[:,3])):
                    if x1 <= leaf_pos*10 <= x2:  # icoord uses 10 units per leaf
                        cluster_nodes.append(j)
            
            # Get the highest merge point for this cluster
            cluster_heights = [dcoord[node][1] for node in cluster_nodes]
            cluster_height = max(cluster_heights) if cluster_heights else max_d
            
            # Calculate cluster center
            cluster_leaves = [R['leaves'].index(idx) for idx in indices]
            x_center = np.mean(cluster_leaves)
            
            # Annotation positioning
            arrow_y = cluster_height * 1.05
            text_y = cluster_height * 1.55
            
            plt.annotate(
                f'Cluster {i+1}', 
                xy=(x_center * (i+1) *5, cluster_height),
                xytext=(x_center * (i+1) * 5, text_y),
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.0',
                    color='darkorange',
                    lw=1.5
                ),
                fontsize=11
            )

    # Adjust y-axis limits for annotations
    max_y = max([max(d) for d in R['dcoord']] + [text_y]) * 1.2
    plt.ylim(top=max_y)
    
    plt.title(f'Hierarchical Clustering of Image Augmentations\n(Identified {num_clusters} main clusters based on CLIP embedding similarity)', 
              fontsize=16, pad=20)
    plt.xlabel('Augmentation Type', fontsize=14, labelpad=15)
    plt.ylabel('Distance (1 - Cosine Similarity)', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # More space for rotated labels
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dendrogram visualization to {output_path}")
    
    return avg_distances, aug_keys

def compute_augmentation_metrics(embedding_pt_path, dataset, num_samples=None):
    """
    Computes metrics using pre-saved embeddings while ensuring image-path alignment.
    
    Args:
        embedding_pt_path (str): Path to embeddings file with image_path entries
        dataset (ImageTransformDataset): Dataset with .from_file() method
        num_samples (int): Optional limit on number of samples to process
        
    Returns:
        pd.DataFrame: Metrics dataframe with augmentation scores
    """
    # Load pre-computed embeddings
    all_embeddings = torch.load(embedding_pt_path)
    if num_samples is not None:
        all_embeddings = all_embeddings[:num_samples]

    # Get augmentation keys from first sample's embeddings
    first_embed = all_embeddings[0]['embeddings']
    aug_keys = [k for k in first_embed.keys() if k != "original"]
    
    # Initialize metrics storage
    metrics = [
        "Embedding Similarity", 
        "Attention Shift",
        "Patch Similarity",
        "Edge Preservation",
        "Detail Preservation"
    ]
    metric_store = {aug: {m: 0 for m in metrics} for aug in aug_keys}
    valid_samples = 0

    for embed_entry in tqdm(all_embeddings, desc="Processing metrics"):
        try:
            # Get original image using dataset's from_file method
            img_entry = dataset.from_file(embed_entry["image_path"])
            
            # Verify augmentation keys match
            if not all(k in img_entry for k in aug_keys):
                missing = [k for k in aug_keys if k not in img_entry]
                print(f"Skipping {embed_entry['image_path']}: Missing augmentations {missing}")
                continue

            # Process each augmentation type
            for aug_key in aug_keys:
                # Get pre-computed embeddings
                orig_emb = np.array(embed_entry['embeddings']['original'])
                aug_emb = np.array(embed_entry['embeddings'][aug_key])
                
                # 1. Embedding Similarity (from pre-computed)
                cos_sim = np.dot(orig_emb.flatten(), aug_emb.flatten())
                cos_sim /= (np.linalg.norm(orig_emb) * np.linalg.norm(aug_emb))
                metric_store[aug_key]["Embedding Similarity"] += cos_sim

                # Get actual images for other metrics
                orig_img = np.array(img_entry["original"])
                aug_img = np.array(img_entry[aug_key])

                # 2. Attention Shift
                orig_attn = get_attention_map(orig_img)
                aug_attn = get_attention_map(aug_img)
                attn_diff = np.mean((orig_attn - aug_attn) ** 2)
                metric_store[aug_key]["Attention Shift"] += 1 / (1 + attn_diff)

                # 3. Patch Similarity
                patch_sim = calculate_patch_similarity(orig_img, aug_img)
                metric_store[aug_key]["Patch Similarity"] += patch_sim

                # 4. Edge Preservation
                edge_sim = calculate_edge_similarity(orig_img, aug_img)
                metric_store[aug_key]["Edge Preservation"] += edge_sim

                # 5. Detail Preservation
                detail_sim = calculate_detail_preservation(orig_img, aug_img)
                metric_store[aug_key]["Detail Preservation"] += detail_sim

            valid_samples += 1

        except Exception as e:
            print(f"Error processing {embed_entry.get('image_path','unknown')}: {str(e)}")
            continue

    # Normalize metrics by number of successfully processed samples
    for aug_key in aug_keys:
        for metric in metrics:
            metric_store[aug_key][metric] /= valid_samples if valid_samples > 0 else 1

    return pd.DataFrame.from_dict(metric_store, orient='index')

# Helper functions for metric calculations
def calculate_patch_similarity(orig, aug, grid_size=4):
    """Calculate patch similarity using MSE-based similarity metric"""
    h, w = orig.shape[:2]
    patch_sim = 0
    for y in range(grid_size):
        for x in range(grid_size):
            # Extract patches
            orig_patch = orig[y*(h//grid_size):(y+1)*(h//grid_size), 
                            x*(w//grid_size):(x+1)*(w//grid_size)]
            aug_patch = aug[y*(h//grid_size):(y+1)*(h//grid_size),
                           x*(w//grid_size):(x+1)*(w//grid_size)]
            
            # Calculate MSE and convert to similarity
            mse = np.mean((orig_patch - aug_patch) ** 2)
            patch_sim += 1 / (1 + mse / 255**2)
    return patch_sim / (grid_size**2)

def calculate_edge_similarity(orig, aug):
    """Calculate edge preservation using gradient-based method"""
    def get_edges(img):
        # Convert to grayscale
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        # Calculate gradients
        dx = np.abs(np.gradient(gray, axis=1))  # Horizontal gradient
        dy = np.abs(np.gradient(gray, axis=0))  # Vertical gradient
        return (dx + dy) / 2  # Combine gradients
    
    orig_edges = get_edges(orig)
    aug_edges = get_edges(aug)
    
    # Normalize edge maps
    orig_edges = orig_edges / orig_edges.max() if orig_edges.max() > 0 else orig_edges
    aug_edges = aug_edges / aug_edges.max() if aug_edges.max() > 0 else aug_edges
    
    return 1 - np.mean(np.abs(orig_edges - aug_edges))

def calculate_detail_preservation(orig, aug, grid_size=8):
    """Calculate detail preservation using patch standard deviation analysis"""
    # Convert to grayscale
    orig_gray = np.dot(orig[...,:3], [0.299, 0.587, 0.114])
    aug_gray = np.dot(aug[...,:3], [0.299, 0.587, 0.114])
    
    h, w = orig_gray.shape
    detail_sim = 0
    valid_patches = 0
    
    for y in range(grid_size):
        for x in range(grid_size):
            y_start = y * (h // grid_size)
            y_end = (y + 1) * (h // grid_size)
            x_start = x * (w // grid_size)
            x_end = (x + 1) * (w // grid_size)
            
            orig_patch = orig_gray[y_start:y_end, x_start:x_end]
            aug_patch = aug_gray[y_start:y_end, x_start:x_end]
            
            # Skip patches with no variation in original
            orig_std = np.std(orig_patch)
            if orig_std < 1e-6:
                continue
                
            # Calculate standard deviation ratio
            aug_std = np.std(aug_patch)
            std_ratio = aug_std / orig_std
            
            # Compute similarity score
            detail_sim += np.exp(-np.abs(np.log(std_ratio)))
            valid_patches += 1
    
    return detail_sim / valid_patches if valid_patches > 0 else 0.0


def plot_augmentation_comparison(metrics_df, output_path="augmentation_comparison.png"):
    """
    Creates a unified horizontal bar chart comparing augmentations across metrics.
    """
    # Get metrics and augmentations
    metrics = metrics_df.index.tolist()
    augmentations = metrics_df.columns.tolist()
    num_metrics = len(metrics)
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # Create color palette
    palette = sns.color_palette("husl", num_metrics)
    metric_colors = {metric: palette[i] for i, metric in enumerate(metrics)}

    # Plot configuration
    bar_height = 0.7
    spacing = 0.3  # Space between metric groups
    
    # Calculate positions
    y_ticks = []
    y_labels = []
    
    # Track best/worst performers per metric
    best_worst = {
        metric: {
            'best': metrics_df.loc[metric].idxmax(),
            'worst': metrics_df.loc[metric].idxmin()
        } for metric in metrics
    }

    # Plot each metric's data
    for metric_idx, metric in enumerate(metrics):
        # Sort augmentations for this metric
        sorted_augs = metrics_df.loc[metric].sort_values(ascending=False)
        
        # Calculate y positions
        y_pos = np.arange(len(sorted_augs)) * (num_metrics + spacing) + metric_idx
        
        # Plot bars
        bars = ax.barh(
            y_pos,
            sorted_augs,
            height=bar_height,
            color=metric_colors[metric],
            edgecolor='none',
            alpha=0.8,
            label=metric
        )
        
        # Add value labels with color coding
        for idx, (aug, value) in enumerate(sorted_augs.items()):
            label_color = 'black'  # Default
            if aug == best_worst[metric]['best']:
                label_color = 'green'
            elif aug == best_worst[metric]['worst']:
                label_color = 'red'
            
            ax.text(
                value + 0.02,
                y_pos[idx],
                f'{value:.2f}',
                va='center',
                color=label_color,
                fontweight='bold' if label_color != 'black' else 'normal'
            )
        
        # Store positions for y-axis labels
        if metric_idx == 0:
            y_ticks.extend(y_pos)
            y_labels.extend([aug.replace('_', ' ').title() for aug in sorted_augs.index])

        # Add average line
        avg = sorted_augs.mean()
        ax.axvline(avg, color=metric_colors[metric], linestyle='--', alpha=0.7)
        ax.text(
            avg, y_pos[-1] + bar_height,
            f'{metric} Avg: {avg:.2f}',
            color=metric_colors[metric],
            va='bottom',
            fontsize=10
        )

    # Configure axis
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Score (Higher is Better)', fontsize=12)
    ax.set_ylabel('Augmentations', fontsize=12)
    
    # Create legend
    handles = [plt.Rectangle((0,0),1,1, color=metric_colors[metric]) for metric in metrics]
    ax.legend(handles, metrics, title='Metrics', 
              bbox_to_anchor=(1.02, 1), 
              loc='lower right')

    # Add titles and explanations
    plt.title('Augmentation Performance Comparison\n(Color = Metric, Text Color = Best/West per Metric)', 
              fontsize=14, pad=20)
    plt.figtext(0.5, 0.01, 
                "Dashed lines show average score for each metric\n"
                "Green text indicates best performer per metric, red indicates worst",
                ha="center", fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved unified comparison to {output_path}")


def plot_metric_heatmap(metrics_df, output_path="metrics_heatmap.png"):
    """
    Creates a heatmap visualization of all metrics and augmentations.
    """
    # Set Seaborn style
    sns.set_style("white")
    
    # Create readable version of data
    plot_df = metrics_df.copy()
    plot_df.index = [idx.replace('_', ' ').title() for idx in plot_df.index]
    plot_df.columns = [col.replace('_', ' ').title() for col in plot_df.columns]

    # Create clustered heatmap
    g = sns.clustermap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        figsize=(14, 14),  # Increased height for better spacing
        dendrogram_ratio=(0.15, 0.15),  # More space for dendrograms
        cbar_pos=(1.02, 0.3, 0.03, 0.4),  # Position colorbar
        tree_kws={'linewidths': 0.5},
        annot_kws={"size": 9},
    )

    # Adjust title positioning
    plt.subplots_adjust(top=0.93)  # Make space above heatmap
    g.fig.suptitle('Augmentation Performance Across All Metrics', 
                  fontsize=16, y=0.97)
    
    # Rotate and align labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    
    # Add description below
    g.fig.text(0.5, 0.01, 
              "This heatmap shows how each augmentation performs across different metrics.\n"
              "Augmentations are clustered based on similar performance patterns.",
              ha="center", fontsize=12, 
              bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8))

    # Save clustered version
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a regular heatmap too (more straightforward)
    plt.figure(figsize=(14, 8))
    
    # Sort columns by average performance
    col_order = plot_df.mean().sort_values(ascending=False).index
    
    # Create a straightforward heatmap 
    ax = sns.heatmap(
        plot_df[col_order],
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Score (higher is better)"}
    )
    
    # Customize appearance
    plt.title('Augmentation Performance Across All Metrics (Sorted by Avg Performance)', fontsize=16)
    plt.ylabel('Metric', fontsize=12)
    plt.xlabel('Augmentation', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = str(output_path)
    plt.savefig(output_path.replace('.png', '_sorted.png'), dpi=300)
    plt.close()
    
    print(f"Saved metrics heatmap to {output_path}")


def plot_augmentation_profile(metrics_df, output_path="augmentation_profiles.png"):
    """
    Creates a panel with individual performance profiles for each augmentation.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics for each augmentation
        output_path (str): Path to save the visualization
    """
    # Get augmentations
    augmentations = metrics_df.columns.tolist()
    metrics = metrics_df.index.tolist()
    
    # Set appealing Seaborn style
    sns.set_style("whitegrid")
    
    # Calculate the grid dimensions
    n_cols = 3
    n_rows = (len(augmentations) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), sharey=True)
    if n_rows == 1:
        axes = [axes]  # Handle the case of a single row
    axes = np.array(axes).flatten()
    
    # Calculate overall average per augmentation for ranking
    avg_performance = metrics_df.mean()
    rank_order = avg_performance.sort_values(ascending=False).index
    rank_dict = {aug: i+1 for i, aug in enumerate(rank_order)}
    
    # Create color palette based on overall performance rank
    norm = plt.Normalize(1, len(augmentations))
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Set a consistent y-limit for all plots
    global_ylim = (0, 1.05)
    
    # Find the best performing augmentation for each metric
    best_per_metric = {metric: metrics_df.loc[metric].idxmax() for metric in metrics}
    
    # For each augmentation, create a performance profile
    for i, aug in enumerate(augmentations):
        ax = axes[i]
        
        # Get this augmentation's values and create a DataFrame for Seaborn
        values = metrics_df[aug]
        data = pd.DataFrame({
            'Metric': metrics,
            'Score': values
        })
        
        # Get rank and color
        rank = rank_dict[aug]
        color = sm.to_rgba(rank)
        
        # Create a Seaborn bar chart
        bars = sns.barplot(
            x='Metric', 
            y='Score',
            data=data,
            ax=ax,
            color=color,
            alpha=0.8,
            edgecolor='gray',
            linewidth=0.8
        )
        
        # Highlight metrics where this augmentation is the best
        for j, metric in enumerate(metrics):
            bar = ax.patches[j]
            if best_per_metric[metric] == aug:
                bar.set_edgecolor('green')
                bar.set_linewidth(2)
                ax.text(
                    j,
                    values[metric] + 0.03,
                    "BEST",
                    ha='center',
                    color='green',
                    fontweight='bold',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="green", alpha=0.7)
                )
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(
                j,
                max(v - 0.08, 0.08),
                f"{v:.2f}",
                ha='center',
                color='white' if v > 0.4 else 'black',
                fontweight='bold',
                fontsize=9
            )
        
        # Make metric names more readable
        ax.set_xticklabels([m.replace('Preservation', 'Pres.').replace('Similarity', 'Sim.') 
                            for m in metrics], rotation=45, ha='right', fontsize=9)
        
        # Add rank
        ax.text(
            0.02, 0.98, 
            f"Rank: #{rank}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontweight='bold',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8)
        )
        
        # Add average score
        avg_score = values.mean()
        ax.text(
            0.98, 0.98, 
            f"Avg: {avg_score:.2f}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8)
        )
        
        # Customize appearance
        ax.set_title(aug.replace('_', ' ').title(), fontsize=12,y=1.1)
        ax.set_ylim(global_ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove some axis labels for cleaner look
        ax.set_ylabel("Score" if i % n_cols == 0 else "")
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Individual Augmentation Performance Profiles', fontsize=18, y=0.99)
    
    # Add a colorbar showing the ranking
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Overall Rank (1=Best)', rotation=270, labelpad=20)
    cbar.set_ticks(np.linspace(1, len(augmentations), min(10, len(augmentations))))
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
                "Each chart shows how an augmentation performs across all metrics.\n"
                "Green highlights indicate metrics where this augmentation performs best among all options.",
                ha="center", fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="skyblue", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved augmentation profiles to {output_path}")


def create_augmentation_radar_chart(metrics_df, output_path="augmentation_radar.png"):
    """
    Creates a grid of radar charts comparing each augmentation against the
    best and worst performers for each individual metric.
    """
    # Get metrics and augmentations
    metrics = metrics_df.index.tolist()
    augmentations = metrics_df.columns.tolist()
    n_metrics = len(metrics)
    n_augs = len(augmentations)
    
    # Calculate best/worst values per metric
    max_values = metrics_df.max(axis=1)
    min_values = metrics_df.min(axis=1)
    
    # Set up radar chart angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure with subplots
    cols = min(4, n_augs)  # Max 4 columns
    rows = int(np.ceil(n_augs / cols))
    fig = plt.figure(figsize=(cols*4, rows*4))
    
    # Create a common legend outside the plots
    fig.legend(handles=[
        plt.Line2D([0], [0], color='#2ecc71', lw=2, label='Best Per Metric'),
        plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Worst Per Metric'),
        plt.Line2D([0], [0], color='#3498db', lw=2, label='Current Augmentation')
    ], loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), frameon=False)

    # Create color maps
    best_color = '#2ecc71'  # Green
    worst_color = '#e74c3c'  # Red
    aug_color = '#3498db'  # Blue

    for idx, aug in enumerate(augmentations):
        ax = fig.add_subplot(rows, cols, idx+1, polar=True)
        
        # Get current augmentation values
        current_values = metrics_df[aug].tolist()
        current_values += current_values[:1]
        
        # Plot best/worst ranges
        ax.fill(angles, max_values.tolist() + [max_values[0]], 
                color=best_color, alpha=0.3, label='_nolegend_')
        ax.fill(angles, min_values.tolist() + [min_values[0]], 
                color=worst_color, alpha=0.3, label='_nolegend_')
        
        # Plot current augmentation
        ax.plot(angles, current_values, color=aug_color, linewidth=2)
        ax.fill(angles, current_values, color=aug_color, alpha=0.2)
        
        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace(' ', '\n') for m in metrics], fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_title(aug.replace('_', ' ').title(), fontsize=10, pad=12)
        
        # Add metric value annotations
        for i, (angle, value) in enumerate(zip(angles[:-1], metrics_df[aug])):
            ax.annotate(f"{value:.2f}", 
                        xy=(angle, value + 0.05),
                        ha='center', va='center',
                        fontsize=7, color=aug_color)

    plt.suptitle("Augmentation Performance Relative to Metric Extremes\n", y=1, fontsize=14)
    plt.figtext(0.8, .01, 
                "Shaded areas show best/worst values for each metric across all augmentations\n"
                "Blue lines show current augmentation's performance", 
                ha="center", fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparative radar grid to {output_path}")


if __name__ == "__main__":
    images_dir = "./dataset"  # Directory with a few sample images
    output_dir = Path("./visualization_output/")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize CLIP model
    clip_processor = CLIPImageProcessor(model_name="openai/clip-vit-base-patch32")
    
    # Create dataset with transformations
    dataset = ImageTransformDataset(
        image_dir=images_dir,
        image_size=(224, 224)
    )
    
    # Generate dendrogram visualization
    avg_distances, aug_keys = plot_augmentation_dendrogram(
        clip_processor=clip_processor,
        dataset=dataset,
        output_path=output_dir / "augmentation_dendrogram.png",
        num_samples=50
    )
    
    # Compute augmentation metrics
    metrics_df = compute_augmentation_metrics(
        embedding_pt_path="./clip_output/clip_embeddings_incremental.pt",
        dataset=dataset,
        num_samples=50
    )

    # Generate augmentation comparison bar charts
    plot_augmentation_comparison(
        metrics_df=metrics_df,
        output_path=output_dir / "augmentation_comparison.png"
    )

    # Generate augmentation heatmap
    plot_metric_heatmap(
        metrics_df=metrics_df,
        output_path=output_dir / "metrics_heatmap.png"
    )

    # Generate augmentation performance profiles
    plot_augmentation_profile(
        metrics_df=metrics_df,
        output_path=output_dir / "augmentation_profiles.png"
    )

    # Generate radar chart visualization
    create_augmentation_radar_chart(
        metrics_df=metrics_df,
        output_path=output_dir / "augmentation_radar.png",
    )
