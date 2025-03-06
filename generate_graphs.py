
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

# Example usage:
if __name__ == "__main__":
    # Directory containing a single image (make sure this directory contains input_image.jpg)
    single_image_dir = "./input_dir/"  
    # plot_image_augmentations(single_image_dir, image_size=(224,224), output_path="visualization.png")
