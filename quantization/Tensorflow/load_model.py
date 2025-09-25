import torch
import torch.nn as nn
from pathlib import Path
import importlib


def load_model_from_pt(pt_path: Path, model_factory: str | None, device: str):
    """
    Tries:
      1) TorchScript (torch.jit.load)
      2) state_dict (requires --model_factory "pkg.mod:factory")
    Returns: (model, is_torchscript)
    """
    try:
        m = torch.jit.load(str(pt_path), map_location=device)
        m.eval()
        print("[Info] Loaded TorchScript model.")
        return m, True
    except Exception as e:
        print(f"[Info] Not a TorchScript model ({e}). Will try state_dict path...")

    if not model_factory:
        raise ValueError("State-dict .pt requires --model_factory 'pkg.mod:factory' to build arch.")

    mod_name, func_name = model_factory.split(":")
    factory = getattr(importlib.import_module(mod_name), func_name)
    model: nn.Module = factory()
    ckpt = torch.load(str(pt_path), map_location=device)

    # Try common keys
    if isinstance(ckpt, dict):
        sd = None
        for k in ("state_dict", "model", "model_state", "state", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
        if sd is None:
            # maybe plain state_dict
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
        if sd is None:
            raise RuntimeError("Could not find state_dict in checkpoint.")
        model.load_state_dict(sd, strict=False)
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.eval()
    print("[Info] Loaded model from state_dict via factory.")
    return model, False