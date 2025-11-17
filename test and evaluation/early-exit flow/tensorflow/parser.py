import numpy as np

def parse_yolo_tflite_identity(identity_tensor: np.ndarray, num_classes: int):
    """
    identity_tensor: np.ndarray with shape (1, 4 + C, 8400).
    Returns:
      boxes:  (8400, 4) in xywh (center) format (as exported by YOLOv8)
      scores: (8400, C) class probabilities in [0,1]
    """
    assert identity_tensor.ndim == 3 and identity_tensor.shape[0] == 1, \
        f"Expected (1, 4+C, 8400), got {identity_tensor.shape}"
    _, nc4, n_anchors = identity_tensor.shape
    assert nc4 == 4 + num_classes, f"Given num_classes={num_classes}, but tensor has {nc4} channels"

    # Split channels: first 4 are boxes, rest are per-class logits
    boxes = identity_tensor[0, :4, :].transpose(1, 0)          # (8400, 4)
    class_logits = identity_tensor[0, 4:, :].transpose(1, 0)   # (8400, C)

    # Ultralytics exports logits; apply sigmoid to get probabilities
    scores = 1.0 / (1.0 + np.exp(-class_logits))
    return boxes, scores
