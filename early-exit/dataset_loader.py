from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class YoloTxtDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and labels in YOLO format.

    This dataset_loader reads images from a given directory and their corresponding
    annotation files in YOLO text format. Each label file contains bounding
    box annotations with normalized coordinates.

    Args:
        img_dir (str or Path): Path to the directory containing image files (.jpg).
        label_dir (str or Path): Path to the directory containing YOLO label files (.txt).
        transform (callable, optional): A torchvision-style transform function to
            apply to images.

    Returns:
        tuple:
            - image (torch.Tensor): The transformed image tensor of shape (C, H, W).
            - labels (torch.Tensor): A tensor of shape (N, 5), where each row 
              corresponds to one object in YOLO format:
                  [class_id, center_x, center_y, width, height]
              Coordinates are normalized to the image dimensions.
    """
    def __init__(self, img_dir, label_dir, transform=None):
        
        # Store the paths to the image and label directories
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        
        # Collect all .jpg image files in the image directory
        self.images = list(self.img_dir.glob("*.jpg"))
        
        # Define the image transformation (default: convert to tensor)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess the image
        # Get the image file path at index idx
        img_path = self.images[idx]
        # Open the image and ensure it's in RGB format
        image = Image.open(img_path).convert("RGB")
        # Apply the given transformation (e.g., convert to tensor, resize, normalize, etc.)
        image = self.transform(image)

        # Load and preprocess the label -> Construct the corresponding label file path (.txt with YOLO format)
        label_path = self.label_dir / (img_path.stem + ".txt")
        
        if label_path.exists():
            # Load the label file as a NumPy array (each row = one bounding box: [class, x, y, w, h])
            # ndmin=2 ensures the result is always 2D, even if only one label exists
            labels = np.loadtxt(label_path, ndmin=2)
        else:
            # If no label file exists, create an empty array with shape (0, 5)
            labels = np.zeros((0, 5), dtype=np.float32)

        # Convert labels from NumPy to a PyTorch tensor
        labels = torch.tensor(labels, dtype=torch.float32)

        # Return the image tensor and its associated labels
        return image, labels