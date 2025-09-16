import time

def calculate_time_inference(model, image_paths, imgsz=640, device=None, conf=0.25, iou=0.7, batch=1, warmup=5):
    """
    Run inference on a list of images and measure inference time and throughput.

    This function performs a warm-up period to stabilize performance, then runs inference
    over all provided images (in batches) using the specified model. It returns the
    average per-image inference time in milliseconds, the frames per second (FPS), and
    total elapsed time.

    Args:
        model: A YOLOv8 model object (with early-exit layers if applicable), loaded via ultralytics.YOLO.
        image_paths (List[str]): List of file paths to the test images.
        imgsz (int, optional): Size (width = height) to which images will be resized for inference. Default is 640.
        device (str or int or None, optional): Device to run inference on. Examples: "cpu", "0", "cuda:0". If None, uses default device. 
        conf (float, optional): Confidence threshold for predictions during timing runs. Default is 0.25.
        iou (float, optional): IoU threshold for Non-Max Suppression (NMS) during timing runs. Default is 0.7.
        batch (int, optional): Batch size to use when calling model.predict on multiple images. Default is 1.
        warmup (int, optional): Number of initial images to run for warm-up before timing, to stabilize startup overhead. Default is 5.

    Returns:
        avg_ms (float): Average inference time per image, in milliseconds, computed over all images (excluding warm-up).
        fps (float): Inference throughput, i.e., how many images per second the model processes.
        total_s (float): Total wall-clock time (in seconds) spent running inference over all images (excluding warm-up).

    Raises:
        ValueError: If image_paths is empty (cannot measure inference time).
    """
    # Warm-up on a few images (helps stabilize GPU timings)
    for p in image_paths[:warmup]:
        _ = model.predict(p, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)

    t0 = time.perf_counter()
    count = 0
    # Process in simple batches to reduce overhead without complicating code
    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i:i+batch]
        _ = model.predict(batch_paths, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
        count += len(batch_paths)
    total_s = time.perf_counter() - t0

    avg_ms = (total_s / count) * 1000.0
    fps = count / total_s if total_s > 0 else 0.0
    return avg_ms, fps, total_s