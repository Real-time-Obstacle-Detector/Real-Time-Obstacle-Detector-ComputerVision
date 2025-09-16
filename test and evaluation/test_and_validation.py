import argparse
import glob
import os
import sys
from pathlib import Path
from time_inference import calculate_time_inference
from accuracy_metrics import calculate_accuracy_metrics
from print_and_save_results import print_and_save_results
from ultralytics import YOLO

def collect_images(test_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    img_dir = test_dir / "images"
    files = []
    for ext in exts:
        files.extend(glob.glob(str(img_dir / f"*{ext}")))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return files

def main():

    #Parse all passed arguments from bash and command line
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n (.pt) on a YOLO-formatted test set and time inference.")
    parser.add_argument("--model", required=True, type=str, help="Path to YOLOv8 .pt weights (your early-exit model).")
    parser.add_argument("--data_yaml", required=True, type=str, help="Path to data.yaml file of your dataset.")
    parser.add_argument("--test_dir", required=True, type=str, help="Folder containing test/images and test/labels")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. 'cpu', '0' or 'cuda:0'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for timing runs (not used in val metrics curves)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS during timing runs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for both val() and timing loop")
    parser.add_argument("--name", type=str, default="eval_run", help="Name of the output subfolder")
    parser.add_argument("--project", type=str, default="runs/eval", help="Project directory to save outputs")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.project) / args.name
    data_yaml = Path(args.data_yaml)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_txt = out_dir / "results.txt"

    #Basic checks for possible problems in our passed arguments
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not (test_dir / "images").exists() or not (test_dir / "labels").exists():
        print(f"ERROR: test_dir must contain 'images/' and 'labels/' subfolders: {test_dir}")
        sys.exit(1)

    # Load model from given model_path
    model = YOLO(str(model_path))

    map_50_95, map_50, precision, recall = calculate_accuracy_metrics(
        data=data_yaml,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
        plots=False
    )

    image_paths = collect_images(test_dir)

    avg_ms, fps, total_s = calculate_time_inference(
        model=model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch
    )

    print_and_save_results(
        model_path= model_path,
        test_dir= test_dir,
        image_paths= image_paths,
        imgsz= args.imgsz,
        device= args.device,
        batch= args.batch,
        map_50_95= map_50_95,
        map_50= map_50,
        precision= precision,
        recall= recall,
        avg_ms= avg_ms,
        fps= fps, 
        total_s= total_s,
        results_txt= results_txt,
        out_dir= out_dir.resolve()
    )

if __name__ == "__main__":
    main()