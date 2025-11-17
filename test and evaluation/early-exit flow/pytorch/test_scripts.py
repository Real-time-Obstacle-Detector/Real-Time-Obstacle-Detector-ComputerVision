from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
from image_loader import build_image_loader
from model_loader import load_model
from counter import ExitCounter

EXIT_SNIPPETS = [
    "model.3.cv3.0.2",    # first-exit
    "model.6.cv3.0.2",    # second-exit
    "model.9.cv3.0.2",    # third-exit
    "model.12.cv3.0.2",   # fourth-exit
    "model.16.cv3.0.2",   # neck-side exit
    "model.26.cv3.2.2"    # final head
]

WEIGHTS     = r"test and evaluation/early-exit flow/pytorch/best.pt"
IMAGES_DIR  = r"C:/Users/abt/Documents/Real-time-obstacle-detector/datasets/dataset/dataset/test/images"
IMG_SIZE    = 640
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THR    = 0.25    # confidence threshold for counting detections
LIMIT_IMGS  = 1500    # set lower for quick runs
REPORT_PATH = "ee_exit_counts.txt"


def run_probe(weights: str, images_dir: str, imgsz: int, device: str,
              exit_snippets: List[str], conf_thr: float, limit_imgs: int, report_path: str):
    
    model, kind = load_model(weights, device)

    print(f"Loaded model ({kind}) on {device}")

    loader, paths = build_image_loader(
        img_dir= images_dir, 
        imgsz=imgsz
    )

    counter = ExitCounter(
        model=model, 
        exit_snippets=exit_snippets,
        conf_thr= conf_thr
    )

    counter.attach()

    with torch.no_grad():
        for i, (p, x) in enumerate(loader()):
            if i >= limit_imgs:
                break

            counter.begin_image()

            x = x.to(device, non_blocking=True)
            
            _ = model(x)  # triggers hooks
            counter.end_image()

    counter.detach()

    report = []
    report.append(f"# Early-Exit Detection Report\n")
    report.append(f"Model: {weights}")
    report.append(f"Device: {device}")
    report.append(f"Images dir: {images_dir}")
    report.append(f"Image size: {imgsz}")
    report.append(f"Confidence threshold: {conf_thr}")
    report.append(f"Images processed: {min(limit_imgs, len(paths))}")
    report.append("")
    
    ordered_exits = counter._ordered_exit_names()
    header = "image_idx\t" + "\t".join(ordered_exits) + "\tTOTAL"
    report.append(header)
    total_per_exit = {k: 0 for k in ordered_exits}
    
    for idx, row in enumerate(counter.per_image_counts):
        total_row = 0
        vals = []
        for k in ordered_exits:
            v = int(row.get(k, 0))
            vals.append(str(v))
            total_per_exit[k] += v
            total_row += v
        report.append(str(idx) + "\t" + "\t".join(vals) + "\t" + str(total_row))
    
    grand = sum(total_per_exit.values())
    report.append("")
    report.append("TOTALS\t" + "\t".join(str(total_per_exit[k]) for k in ordered_exits) + "\t" + str(grand))
    report_text = "\n".join(report)
    Path(report_path).write_text(report_text, encoding="utf-8")
    print(f"\nSaved report to: {Path(report_path).resolve()}")

if __name__ == "__main__":
    run_probe(
        WEIGHTS, IMAGES_DIR, IMG_SIZE, DEVICE,
        EXIT_SNIPPETS, CONF_THR, LIMIT_IMGS, REPORT_PATH
    )
