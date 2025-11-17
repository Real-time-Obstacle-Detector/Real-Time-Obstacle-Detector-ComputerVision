from parser import parse_yolo_tflite_identity

def group_detection_outputs(out_details, interpreter, num_classes: int):
    """
    Returns a dict exit_name -> {'boxes':..., 'scores':...}
    Works for:
      - Single head: one (1, 4+C, 8400) tensor named 'Identity'
      - Multi-exit: multiple such tensors (Identity, Identity_1, ...)
    """
    candidates = []
    for od in out_details:
        # Pull tensor to inspect its shape
        arr = interpreter.get_tensor(od['index'])
        print(od['name'], arr.shape)
        if (arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 8400 and arr.shape[1] >= 1):
            candidates.append((od, arr))

    # No valid detection tensors found → return empty dict
    if not candidates:
        return {}

    # Sort by output index so earlier exits come first (adjust if you have real names)
    candidates.sort(key=lambda x: x[0]['index'])

    outputs = {}
    for i, (od, arr) in enumerate(candidates):
        exit_name = f"ee{i+1}" if len(candidates) > 1 else "final"
        # Infer classes from channel count if not given
        c_tot = arr.shape[1]
        c_guess = c_tot - 4
        c_use = num_classes if (4 + num_classes) == c_tot else c_guess
        boxes, scores = parse_yolo_tflite_identity(arr, c_use)
        outputs[exit_name] = {'boxes': boxes, 'scores': scores}
    return outputs