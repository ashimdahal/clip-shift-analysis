import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import argparse
import json
from pathlib import Path

class CLIPImageProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """Initialize CLIP model and processor.
        
        Args:
            model_name (str): Name of the CLIP model to use
            device (str, optional): Device to run model on. If None, will use CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embeddings(self, images):
        """Get CLIP embeddings for a batch of images.
        
        Args:
            images (list): List of PIL images or numpy arrays
            
        Returns:
            torch.Tensor: Image embeddings (B, embedding_dim)
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_image_features(**inputs)
            return embeddings.cpu()

class ImageTransformDataset(Dataset):
    def __init__(self, image_dir, transforms_dict=None, image_size=(224, 224)):
        """Dataset for applying multiple transformations to images.
        
        Args:
            image_dir (str): Directory containing images
            transforms_dict (dict): Dictionary of named transformations
            image_size (tuple): Size to resize images to
        """
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("**/*.jpg")) + \
                           list(self.image_dir.glob("**/*.jpeg")) + \
                           list(self.image_dir.glob("**/*.png"))
        
        # Basic resize transform that will always be applied first
        self.base_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
        ])
        
        # Default transformations if none provided
        if transforms_dict is None:
            self.transforms_dict = {
                "noise": A.GaussNoise(std_range=(0.44,0.88), p=1.0),
                "blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                "color_jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                "shift_scale_rotate": A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=1.0),
                "horizontal_flip": A.HorizontalFlip(p=1.0),
                "elastic": A.ElasticTransform(p=1.0, alpha=30, sigma=60),
                "perspective": A.Perspective(scale=(0.05, 0.1), p=1.0),
                "random_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                "coarse_dropout": A.CoarseDropout(p=1, num_holes_range=(6,8), fill="random", hole_height_range=(16,32), hole_width_range=(16,32),)
            }
        else:
            self.transforms_dict = transforms_dict
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        
        # Read image as RGB numpy array
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Apply base transformation to ensure correct size and normalization
        original = self.base_transform(image=image)["image"]
        
        # Store original image and transformations
        result = {
            "image_path": image_path,
            "original": self.processor_ready(original)
        }
        
        # Apply each transformation
        for name, transform in self.transforms_dict.items():
            # Apply the transformation, then the base resize/normalize
            transformed = transform(image=image)["image"]
            normalized = self.base_transform(image=transformed)["image"]
            result[name] = self.processor_ready(normalized)
            
        return result
    
    def processor_ready(self, img_array):
        """Convert normalized numpy array to PIL Image for CLIP processor"""
        # Denormalize
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img_denorm = (img_array * std + mean) * 255
        # img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

def custom_collate(batch):
    """Collate a batch of samples into a single dictionary with lists."""
    result = {}
    for key in batch[0].keys():
        result[key] = [item[key] for item in batch]
    return result

def process_images(args):
    """Process images with transformations and save CLIP embeddings."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize CLIP processor
    clip_processor = CLIPImageProcessor(model_name=args.model_name)
    
    # Create dataset and dataloader
    dataset = ImageTransformDataset(
        args.image_dir, 
        image_size=(args.image_size, args.image_size)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )
    
    # Process images and store results
    results = []
    
    for batch in tqdm(dataloader, desc="Processing images"):
        batch_results = []
        
        # Get embeddings for original and each transformation
        for key in batch:
            if key == "image_path":
                continue
                
            # Get embeddings for this transformation
            embeddings = clip_processor.get_embeddings(batch[key])
            
            # Store embeddings with their transformation type
            for i, path in enumerate(batch["image_path"]):
                if len(batch_results) <= i:
                    batch_results.append({"image_path": path, "embeddings": {}})
                batch_results[i]["embeddings"][key] = embeddings[i].tolist()
        
        # Add batch results to overall results
        results.extend(batch_results)
        
        # Save incremental results (in case of crashes)
        if args.save_incremental:
            torch.save(results, output_dir / "clip_embeddings_incremental.pt")
    
    # Save final results in multiple formats
    embedding_path = output_dir / "clip_embeddings.pt"
    json_path = output_dir / "clip_embeddings.json"
    
    # Save as PyTorch file (preserves tensors)
    torch.save(results, embedding_path)
    
    # Save as JSON for easier inspection
    with open(json_path, 'w') as f:
        json.dump(results, f)
        
    print(f"Saved {len(results)} processed images to:")
    print(f"  - {embedding_path} (PyTorch format)")
    print(f"  - {json_path} (JSON format)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with CLIP and transformations")
    parser.add_argument("--image_dir", type=str, default="./dataset/", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./clip_output", help="Output directory")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for CLIP")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--save_incremental", action="store_true", help="Save results incrementally")
    
    args = parser.parse_args()
    process_images(args)
