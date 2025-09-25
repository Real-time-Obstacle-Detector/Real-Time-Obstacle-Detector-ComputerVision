import cv2, glob, random
from pathlib import Path


def build_rep_dataset(calib_dir: Path, imgsz: int, num: int):
    """
    Representative dataset generator for INT8 PTQ.
    Yields NHWC float32 in [0,1].
    """
    
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths += glob.glob(str(calib_dir / ext))
    if not paths:
        raise ValueError(f"No images found in {calib_dir}")

    random.shuffle(paths)
    paths = paths[: max(1, num)]

    def gen():
        for p in paths:
            img = cv2.imread(p)
            if img is None: 
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            img = img.astype("float32") / 255.0
            # NHWC
            yield [img[None, ...]]
    return gen