import numpy as np

def which_exit_fired(interpreter_outputs, per_exit_threshold = 0.3):
    """Determine which early-exit (EE) branch "won" for a single inference result.

    This function inspects the provided interpreter_outputs for a single example and
    selects the earliest EE whose highest class confidence meets or exceeds a
    predefined threshold. It checks exits in the order: "ee1", "ee2", "ee3",
    "ee4", "ee_neck". If an eligible early exit is found, returns its name and the
    maximum score observed for that exit. If no early exit meets its threshold,
    the function falls back to returning "final" and the maximum score from the
    last available output entry.

    Parameters
    ----------
    interpreter_outputs : dict
        Mapping from exit name (str) to a dict-like object containing at least:
          - 'scores': array-like of class confidence scores for that exit
          - 'boxes' : (optional) bounding box predictions (not used by this routine)
        Example shape for 'scores' is typically a 1-D array of class confidences.

    Returns
    -------
    tuple[str, float]
        A pair (exit_name, score) where exit_name is the chosen exit label
        (one of "ee1","ee2","ee3","ee4","ee_neck", or "final") and score is the
        corresponding maximum confidence converted to float.

    """
    
    keys = sorted(interpreter_outputs.keys(),
                  key=lambda k: (k != "final", k))  # 'ee*' first, then 'final'
    print(f"Available exits: {keys}")
    for k in keys:
        scores = interpreter_outputs[k]['scores']   # (8400, C)
        max_conf = float(np.max(scores)) if scores.size else 0.0
        if k != "final" and max_conf >= per_exit_threshold:
            return k, max_conf
    # fallback to final (or highest among provided)
    k = "final" if "final" in interpreter_outputs else keys[-1]
    scores = interpreter_outputs[k]['scores']
    return k, float(np.max(scores)) if scores.size else 0.0