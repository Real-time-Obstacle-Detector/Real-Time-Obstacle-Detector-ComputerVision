import collections
import numpy as np

def count_params_from_interpreter(tensor_details, const_flags):
    """
    Counts and analyzes parameters from a TensorFlow Lite interpreter tensor details.

    This function analyzes tensor details from a TFLite interpreter to count parameters
    and gather statistics about quantized vs float tensors.

    Args:
        tensor_details (list): List of tensor detail dictionaries from TFLite interpreter
        const_flags (dict): Dictionary mapping tensor indices to boolean flags indicating if tensor is constant

    Returns:
        dict: Dictionary containing:
            - const_param_total (int): Total number of constant parameters
            - const_params_by_dtype (dict): Parameters counts broken down by data type
            - num_quantized_tensors (int): Number of quantized tensors
            - num_float_tensors (int): Number of floating point tensors
            - quantized_tensor_scales (list): Preview of quantization scales for first 20 tensors,
              as tuples of (tensor_index, tensor_name, scale_values)
    """
    total = 0
    by_dtype = {}
    quantized_tensors = 0
    float_tensors = 0
    q_scales_preview = []

    def _numel(shape):
        if shape is None:
            return 1
        n = 1
        for d in shape:
            if d is None or int(d) < 0:
                return None
            n *= int(d)
        return n

    for t in tensor_details:
        # shape can be 'shape_signature' or 'shape'
        shape = t.get('shape_signature')
        if shape is None or any(int(d) < 0 for d in shape):
            shape = t.get('shape')
        shape = list(shape) if shape is not None else None
        numel = _numel(shape)

        # dtype comes back as a numpy dtype
        dtype_str = str(t.get('dtype')).upper()

        # quant params are numpy arrays (possibly empty)
        q = t.get('quantization_parameters', {}) or {}
        scales = np.asarray(q.get('scales', []))
        zps    = np.asarray(q.get('zero_points', []))

        # count weights (constants) only
        if numel is not None and const_flags.get(t['index'], False):
            total += numel
            by_dtype[dtype_str] = by_dtype.get(dtype_str, 0) + numel

        # classify quantized vs float by presence of non-empty scales (or int dtype)
        is_quantized = (scales.size > 0) or ('INT' in dtype_str)
        if is_quantized:
            quantized_tensors += 1
            if len(q_scales_preview) < 20:
                q_scales_preview.append((t['index'], t['name'], scales.tolist()))
        else:
            float_tensors += 1

    return {
        'const_param_total': total,
        'const_params_by_dtype': by_dtype,
        'num_quantized_tensors': quantized_tensors,
        'num_float_tensors': float_tensors,
        'quantized_tensor_scales': q_scales_preview,
    }

def diff(a, b):
    out = {}
    for k in a.keys() | b.keys():
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out[k] = diff(va, vb)
        elif isinstance(va, (int,float)) and isinstance(vb, (int,float)):
            out[k] = vb - va
        else:
            out[k] = {'baseline': va, 'compare': vb}
    return out