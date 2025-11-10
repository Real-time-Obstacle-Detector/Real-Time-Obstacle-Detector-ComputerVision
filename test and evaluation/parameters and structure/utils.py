import collections

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
    by_dtype = collections.Counter()
    quantized_tensors = 0
    float_tensors = 0
    q_scales = []
    for t in tensor_details:
        shape = t['shape_signature'] if -1 not in t['shape'] else t['shape_signature']
        if shape is None or len(shape) == 0:
            numel = 1
        else:
            numel = 1
            for d in shape:
                if d < 0:  # unknown
                    numel = None
                    break
                numel *= d
        q = t.get('quantization_parameters', {})
        scale = q.get('scales', [])
        zp = q.get('zero_points', [])
        dtype = str(t['dtype']).upper()
        if numel is not None and const_flags.get(t['index'], False):
            total += numel
            by_dtype[dtype] += numel
        if scale:
            quantized_tensors += 1
            q_scales.append((t['index'], t['name'], list(scale)))
        else:
            float_tensors += 1
    return {
        'const_param_total': total,
        'const_params_by_dtype': dict(by_dtype),
        'num_quantized_tensors': quantized_tensors,
        'num_float_tensors': float_tensors,
        'quantized_tensor_scales': q_scales[:20],  # preview first 20
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