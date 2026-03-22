#!/usr/bin/env python3
"""Generate sample data for Edge Vision Applications testing."""

import argparse
import logging
import os
import numpy as np
from PIL import Image
import cv2

from src.utils import set_deterministic_seed, setup_logging


def generate_sample_images(output_dir: str, num_images: int = 10) -> None:
    """Generate sample images for testing.
    
    Args:
        output_dir: Output directory for sample images
        num_images: Number of images to generate
    """
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating {num_images} sample images in {output_dir}")
    
    for i in range(num_images):
        # Generate random image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some patterns to make it more interesting
        if i % 3 == 0:
            # Add circles
            cv2.circle(image_array, (112, 112), 50, (255, 0, 0), -1)
        elif i % 3 == 1:
            # Add rectangles
            cv2.rectangle(image_array, (50, 50), (174, 174), (0, 255, 0), -1)
        else:
            # Add lines
            cv2.line(image_array, (0, 0), (223, 223), (0, 0, 255), 5)
        
        # Convert to PIL Image and save
        image = Image.fromarray(image_array)
        image_path = os.path.join(output_dir, f"sample_{i:03d}.jpg")
        image.save(image_path)
        
        if (i + 1) % 5 == 0:
            logger.info(f"Generated {i + 1}/{num_images} images")
    
    logger.info(f"Sample images generated successfully")


def generate_synthetic_dataset(output_dir: str, 
                              num_classes: int = 10,
                              samples_per_class: int = 100) -> None:
    """Generate synthetic dataset for testing.
    
    Args:
        output_dir: Output directory for dataset
        num_classes: Number of classes to generate
        samples_per_class: Number of samples per class
    """
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating synthetic dataset with {num_classes} classes, "
               f"{samples_per_class} samples per class")
    
    for class_id in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{class_id:03d}")
        os.makedirs(class_dir, exist_ok=True)
        
        for sample_id in range(samples_per_class):
            # Generate image with class-specific characteristics
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add class-specific patterns
            if class_id % 4 == 0:
                # Horizontal stripes
                for y in range(0, 224, 20):
                    image_array[y:y+10, :] = [class_id * 25, 100, 200]
            elif class_id % 4 == 1:
                # Vertical stripes
                for x in range(0, 224, 20):
                    image_array[:, x:x+10] = [100, class_id * 25, 200]
            elif class_id % 4 == 2:
                # Diagonal pattern
                for i in range(224):
                    for j in range(224):
                        if (i + j) % 20 < 10:
                            image_array[i, j] = [200, 100, class_id * 25]
            else:
                # Random noise with class-specific color bias
                noise = np.random.randint(0, 100, (224, 224, 3))
                image_array = (image_array + noise) % 255
                image_array[:, :, class_id % 3] = np.clip(
                    image_array[:, :, class_id % 3] + 50, 0, 255
                )
            
            # Save image
            image = Image.fromarray(image_array)
            image_path = os.path.join(class_dir, f"sample_{sample_id:04d}.jpg")
            image.save(image_path)
        
        logger.info(f"Generated class {class_id} ({samples_per_class} samples)")
    
    logger.info("Synthetic dataset generation completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate sample data for Edge Vision Applications")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/samples",
        help="Output directory for sample data"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of sample images to generate"
    )
    
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Generate full synthetic dataset"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes for synthetic dataset"
    )
    
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of samples per class"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    if args.dataset:
        # Generate full synthetic dataset
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class
        )
    else:
        # Generate simple sample images
        generate_sample_images(
            output_dir=args.output_dir,
            num_images=args.num_images
        )
    
    logger.info("Sample data generation completed")


if __name__ == "__main__":
    main()
