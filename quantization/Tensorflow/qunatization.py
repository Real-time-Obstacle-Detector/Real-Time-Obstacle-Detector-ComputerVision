"""
Convert a PyTorch .pt (URL) to TFLite FP16 and INT8 (PTQ) WITHOUT Ultralytics.
Pipeline: PyTorch (.pt) -> ONNX -> TensorFlow SavedModel -> TFLite

Notes
- If your .pt is TorchScript, you can omit --model_factory; the script will try torch.jit.load().
- If your .pt is a state_dict, provide --model_factory "module.submodule:factory"
  where `factory()` returns an *uninitialized* nn.Module with the correct architecture.
- INT8 PTQ needs --calib_dir (images) to build a representative dataset.
- TFLite expects NHWC; onnx2tf handles NCHW->NHWC conversion.

"""

import argparse, tempfile, urllib.request
from pathlib import Path

from format_converters import *
from load_model import load_model_from_pt

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_url", required=True, help="URL to .pt checkpoint")
    ap.add_argument("--model_factory", default=None,
                    help="Optional 'pkg.mod:factory' that returns nn.Module (for state_dict .pt). Omit for TorchScript .pt.")
    ap.add_argument("--device", default="cuda:0", help='cuda:0 or cpu')
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out_dir", default="./exports")
    ap.add_argument("--calib_dir", default=None, help="Folder of images for INT8 calibration (NHWC)")
    ap.add_argument("--num_calib", type=int, default=500, help="Calibration image count")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory()
    pt_path = Path(tmp.name) / "model.pt"
    print("[Step] Downloading .pt ...")
    download(args.pt_url, pt_path)
    print(f"[OK] {pt_path}")

    print("[Step] Loading model ...")
    model, is_ts = load_model_from_pt(pt_path, args.model_factory, args.device)

    onnx_path = out_dir / "model.onnx"
    covert_to_onnx(model, is_ts, onnx_path, args.imgsz, args.device)

    savedmodel_dir = out_dir / "saved_model"
    convert_onnx_to_tf_savedmodel(onnx_path, savedmodel_dir)

    convert_tf_to_tflite(savedmodel_dir, out_dir, args.imgsz, args.calib_dir, args.num_calib)
    print("\n[Done] Outputs in:", out_dir.resolve())

if __name__ == "__main__":
    main()