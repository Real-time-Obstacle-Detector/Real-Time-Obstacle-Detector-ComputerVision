import os, glob
import torchvision.transforms as T
from PIL import Image

def build_image_loader(img_dir: str, imgsz: int = 640, exts=(".jpg", ".jpeg", ".png")):
    """
    Builds an image loader generator and returns image paths.

    Args:
        img_dir (str): Directory containing images.
        imgsz (int): Size to which images are resized (imgsz x imgsz).
        exts (tuple): Allowed image file extensions.

    Returns:
        gen (generator): Generator yielding (path, tensor) tuples.
        paths (list): List of image file paths.
    """

    # gather image paths
    paths = sorted([p for ext in exts for p in glob.glob(os.path.join(img_dir, f"*{ext}"))])

    if not paths:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    
    # define transformer
    transformer = T.Compose([
        T.Resize(
            size= (imgsz, imgsz),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.ToTensor(),   # -> [0,1], (C,H,W)
    ])

    def gen():
        for path in paths:
            img = Image.open(path).convert("RGB")
            yield path, transformer(img).unsqueeze(0)

    return gen, paths