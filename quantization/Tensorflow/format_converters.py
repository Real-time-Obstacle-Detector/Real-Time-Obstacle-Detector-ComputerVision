from pathlib import Path
import onnx
from onnxsim import simplify
import subprocess, sys
import torch
import torch.nn as nn
import tensorflow as tf


def covert_to_onnx(model, is_torchscript: bool, onnx_path: Path, imgsz: int, device: str):
    
    model_device = torch.device(device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Example input in NCHW for export
    sample = torch.randn(1, 3, imgsz, imgsz, device=model_device)

    # TorchScript can be exported via torch.onnx.export the same way
    print("[Step] Exporting to ONNX...")
    torch.onnx.export(
        model,
        sample,
        str(onnx_path),
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX saved: {onnx_path}")

    # Optional: simplify
    try:
        
        print("[Step] Simplifying ONNX...")
        sm, ok = simplify(onnx_model)
        if ok:
            onnx.save(sm, str(onnx_path))
            print("[OK] ONNX simplified.")
        else:
            print("[Warn] onnxsim simplify returned False; keeping original.")
    except Exception as e:
        print(f"[Warn] ONNX simplification skipped: {e}")

def convert_onnx_to_tf_savedmodel(onnx_path: Path, savedmodel_dir: Path):
    """
    Use onnx2tf for robust NCHW->NHWC conversion and better TFLite-compat graphs.
    """
    print("[Step] Converting ONNX -> TensorFlow SavedModel (onnx2tf)...")
    # onnx2tf has a CLI; invoke via subprocess for simplicity.
    # Minimal flags; add --output_signaturedefs if you need tf-serving signatures.
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(savedmodel_dir),
        "--output_saved_model"
    ]
    # Disable unnecessary fusions that sometimes hurt custom heads
    # (leave defaults if you like)
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[OK] SavedModel at: {savedmodel_dir}")

def convert_tf_to_tflite(savedmodel_dir: Path, out_dir: Path, imgsz: int, calib_dir: Path | None, num_calib: int):

    out_dir.mkdir(parents=True, exist_ok=True)

    # FP16
    print("[Step] Converting to TFLite FP16...")
    conv = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]  # FP16
    tflite_fp16 = conv.convert()
    (out_dir / "model_fp16.tflite").write_bytes(tflite_fp16)
    print(f"[OK] FP16 TFLite: {out_dir / 'model_fp16.tflite'}")

    # INT8 PTQ (full-integer)
    if calib_dir:
        print("[Step] Converting to TFLite INT8 (PTQ, full-integer)...")
        conv = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = build_rep_dataset(Path(calib_dir), imgsz, num_calib)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tflite_int8 = conv.convert()
        (out_dir / "model_int8.tflite").write_bytes(tflite_int8)
        print(f"[OK] INT8 TFLite: {out_dir / 'model_int8.tflite'}")
    else:
        print("[Info] Skipping INT8 (no --calib_dir provided).")