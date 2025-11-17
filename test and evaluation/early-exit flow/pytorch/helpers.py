import torch
from typing import Dict, List, Tuple, Optional, Union

TensorLike = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict]

def _sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    # If values already in [0,1], leave them; else apply sigmoid.
    if x.min().item() >= 0.0 and x.max().item() <= 1.0:
        return x
    return x.sigmoid()

def _count_from_tensor(t: torch.Tensor, conf_thr: float) -> Optional[int]:
    '''
    Try to infer #detections from a head-like tensor.
    Common raw head: (B, N, 5+C) where [:, :,0:4]=boxes, [:, :, 4]=obj, [:, :, 5:]=cls logits or probs.
    Returns int or None if it doesn't look like a detection tensor.
    
    Args:
        t (torch.Tensor): Tensor to analyze.
        conf_thr (float): Confidence threshold for counting detections.
    Returns:
        int or None: Number of detections inferred, or None if not applicable.
    '''

    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim != 3:
        return None  # most raw heads are (B, N, no)

    _, _, D = t.shape
    if D < 6:
        return None

    obj = _sigmoid_safe(t[..., 4])

    cls_logits = t[..., 5:]
    cls_probs = _sigmoid_safe(cls_logits)
    top_cls, _ = cls_probs.max(dim=-1)
    score = obj * top_cls

    print("scores: ", score, t)

    return int((score >= conf_thr).sum().item())

def _extract_first_tensor(x: TensorLike) -> Optional[torch.Tensor]:
    '''
    Extracts the first torch.Tensor found in a TensorLike structure.
    Args:
        x (TensorLike): Input which may be a tensor, list, tuple, or dict.
    Returns:
        torch.Tensor or None: The first tensor found, or None if none exists.
    '''

    if isinstance(x, torch.Tensor):
        return x
    
    if isinstance(x, (list, tuple)):
        for y in x:
            if isinstance(y, torch.Tensor):
                return y
            
    if isinstance(x, dict):
        for y in x.values():
            if isinstance(y, torch.Tensor):
                return y
            if isinstance(y, (list, tuple)) and y and isinstance(y[0], torch.Tensor):
                return y[0]
            
    return None