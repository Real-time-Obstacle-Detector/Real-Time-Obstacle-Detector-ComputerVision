from datetime import datetime

def print_and_save_results(model_path, test_dir, image_paths, imgsz, device, batch, 
                           map_50_95, map_50, precision, recall, avg_ms,fps, total_s, results_txt,out_dir):
    """
    Print evaluation results to console and save them to a text file.

    This function formats the evaluation metrics (accuracy and speed) along with
    model and test settings into human-readable lines, prints them, and writes
    the same content to a specified .txt file.

    Args:
        model_path (str or Path): Path to the model file that was evaluated.
        test_dir (str or Path): Directory containing the test dataset (images + labels).
        image_paths (List[str] or List[Path]): List of file paths of images that were evaluated.
        imgsz (int): Input image size used during evaluation (width = height).
        device (str or None): Device identifier used for running the evaluation. E.g. "cpu", "cuda:0", or similar.
        batch (int): Batch size used during inference / evaluation.
        map_50_95 (float): Mean Average Precision over IoU thresholds from 0.50 to 0.95.
        map_50 (float): Mean Average Precision at IoU = 0.50.
        precision (float): Mean precision over all classes.
        recall (float): Mean recall over all classes.
        avg_ms (float): Average inference time per image in milliseconds
        fps (float): Frames per second—how many images are processed per second on average.
        total_s (float): Total wall-clock time in seconds for the full set of images processed.
        results_txt (str or Path): Path of the .txt file where results will be saved.
        out_dir (str or Path): Directory where project outputs are stored; used in print messages.
    """

    lines = []
    lines.append(f"YOLOv8 Evaluation — {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Model: {model_path}")
    lines.append(f"Test dir: {test_dir}  (#images: {len(image_paths)})")
    lines.append(f"Image size: {imgsz} | Device: {device or 'auto'} | Batch: {batch}")
    lines.append("")
    lines.append("== Accuracy (Ultralytics val on test split) ==")
    lines.append(f"mAP50-95: {map_50_95:.4f}")
    lines.append(f"mAP50:    {map_50:.4f}")

    if not (precision != precision):
        lines.append(f"Precision (mean): {precision:.4f}")
    if not (recall != recall):
        lines.append(f"Recall (mean):    {recall:.4f}")

    lines.append("")
    lines.append("== Speed ==")
    lines.append(f"Average inference time per image: {avg_ms:.2f} ms")
    lines.append(f"Throughput (FPS): {fps:.2f}")
    lines.append(f"Total wall time on {len(image_paths)} images: {total_s:.2f} s")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved results to: {results_txt}")
    print(f"(Project outputs are save to: {out_dir})")