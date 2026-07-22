#!/usr/bin/env python3
"""
Download and process Brain MRI dataset from Kaggle
and save it as .npz files in datasets_all folder
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import zipfile
import subprocess
from tqdm import tqdm

# Configuration
DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ALL_DIR = os.environ.get(
    "OUT_DIR",
    str(PROJECT_ROOT / "datasets_all" / "hardbench" / "BTMRI"),
)
TEMP_DOWNLOAD_DIR = os.environ.get(
    "TMP_DIR",
    str(PROJECT_ROOT / ".cache" / "brain_mri_raw"),
)
TARGET_SIZE = 224  # Only save 224x224 images

def setup_kaggle():
    """Check if kaggle API is set up"""
    kaggle_config = os.path.expanduser(os.environ.get("KAGGLE_CONFIG", "~/.kaggle/kaggle.json"))
    if not os.path.exists(kaggle_config):
        print("ERROR: Kaggle API credentials not found!")
        print("Please set up kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create an API token")
        print("3. Place it at ~/.kaggle/kaggle.json")
        print("4. chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True

def download_dataset():
    """Download dataset from Kaggle"""
    print(f"Downloading Brain MRI dataset from Kaggle: {DATASET_NAME}")
    
    # Create temp directory
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    
    try:
        # Download using kaggle API
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", TEMP_DOWNLOAD_DIR],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Download successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ kaggle CLI not found. Install with: pip install kaggle")
        return False

def extract_dataset():
    """Extract downloaded zip file"""
    print("Extracting dataset...")
    zip_files = list(Path(TEMP_DOWNLOAD_DIR).glob("*.zip"))
    
    if not zip_files:
        print("✗ No zip files found in download directory")
        return False
    
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DOWNLOAD_DIR)
        os.remove(zip_file)
    
    print("✓ Extraction complete!")
    return True

def load_images_from_folder(folder_path, target_size=224, max_images=None):
    """Load all images from a folder recursively and resize to target size"""
    images = []
    labels = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize immediately to ensure consistent shape
                    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    images.append(np.array(img_resized))
                    
                    # Label based on folder structure (e.g., tumor/no_tumor)
                    folder_name = os.path.basename(root)
                    labels.append(folder_name)
                    
                    if max_images and len(images) >= max_images:
                        return np.array(images), np.array(labels)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def resize_images(images, target_size):
    """Resize images to target size"""
    resized = []
    for img in tqdm(images, desc=f"Resizing to {target_size}x{target_size}"):
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        
        resized_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        resized.append(np.array(resized_img))
    
    return np.array(resized)

def process_and_save():
    """Process images and save as NPZ files"""
    print("Processing Brain MRI dataset...")
    
    # Find the main data directory
    data_base_path = TEMP_DOWNLOAD_DIR
    # Look for testing data only
    possible_paths = [
        os.path.join(data_base_path, "Testing"),
        os.path.join(data_base_path, "test"),
        os.path.join(data_base_path, "brain_tumor_dataset", "Testing"),
        os.path.join(data_base_path, "brain_tumor_dataset", "test"),
        os.path.join(data_base_path, "brain_tumor_dataset"),
        data_base_path
    ]
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            print(f"Found dataset directory: {data_path}")
            break
    
    if not data_path:
        print("✗ Could not find dataset files")
        print(f"Available paths in {data_base_path}:")
        for item in os.listdir(data_base_path):
            print(f"  - {item}")
        return False
    
    # Load and resize images to 224x224
    print(f"Loading and resizing images to {TARGET_SIZE}x{TARGET_SIZE}...")
    images, labels = load_images_from_folder(data_path, target_size=TARGET_SIZE)
    
    if len(images) == 0:
        print("✗ No images found in dataset")
        return False
    
    print(f"✓ Loaded and resized {len(images)} images")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    
    # Create output directory
    os.makedirs(DATASETS_ALL_DIR, exist_ok=True)
    
    # Save 224x224 dataset
    print(f"Saving dataset...")
    save_path = os.path.join(DATASETS_ALL_DIR, "brain_mri_224.npz")
    np.savez(save_path, images=images, labels=labels)
    print(f"✓ Saved to {save_path}")
    
    return True

def cleanup():
    """Remove temporary files"""
    print("Cleaning up temporary files...")
    if os.path.exists(TEMP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
    print("✓ Cleanup complete!")

def main():
    """Main execution"""
    print("=" * 60)
    print("Brain MRI Dataset Download & Processing")
    print("=" * 60)
    
    # Check kaggle setup
    if not setup_kaggle():
        return False
    
    # Download
    if not download_dataset():
        return False
    
    # Extract
    if not extract_dataset():
        return False
    
    # Process and save
    if not process_and_save():
        return False
    
    # Cleanup
    cleanup()
    
    print("=" * 60)
    print(f"✓ Brain MRI test dataset ready in {DATASETS_ALL_DIR}/")
    print("  - brain_mri_224.npz (224x224)")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()
