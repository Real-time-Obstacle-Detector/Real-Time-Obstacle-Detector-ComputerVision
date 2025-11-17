import torch
import torch.nn as nn

def load_model(weights: str, device: torch.device):
    """Loads a PyTorch model from a .pt file.
    Args:
        weights (str): Path to the .pt model file.
        device (torch.device): Device to load the model onto.
    Returns:
        model (nn.Module): Loaded PyTorch model in eval mode.
        kind (str): Kind of model loader used ("ultralytics" or "pytorch").
    """

    # Try Ultralytics first (gives us y.model = nn.Module). If that fails, fall back to torch.load with safe_globals allowlist.
    try:

        from ultralytics import YOLO

        y = YOLO(weights)
        model = y.model.eval().to(device)

        return model, "ultralytics"
    
    except Exception as error:
        print(f"Ultralytics model load failed: {error}. Falling back to raw torch.load...")
        pass

    # Raw torch.load with safe allowlist for custom classes 
    try:
        from torch.serialization import safe_globals
        import ultralytics.nn.tasks as utasks
        
        allow = []
        
        try:
            # If Ultralytics is installed, allow their class names
            allow.append(utasks.DetectionModel)
        except Exception:
            pass
        with safe_globals(allow):
            obj = torch.load(weights, map_location=device, weights_only=False)
    
    except TypeError:

        print("torch.load with safe_globals failed, trying torch,load without safe globs...")
        obj = torch.load(weights, map_location=device)

    if isinstance(obj, nn.Module):
        return obj.eval().to(device), "pytorch"
    
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        return obj["model"].eval().to(device), "pytorch"
    
    raise RuntimeError("Failed to load model from .pt. Install ultralytics or save a plain nn.Module checkpoint.")