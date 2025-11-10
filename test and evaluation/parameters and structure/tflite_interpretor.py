import tensorflow as tf

def interpreter_details(model_path):
    """
    Analyze a TensorFlow Lite model and return its tensor details along with a heuristic
    map indicating which tensors are likely constants (model buffers) versus runtime tensors.

    Args:
        model_path (str): Filesystem path to a .tflite model file. This function will construct
            a tf.lite.Interpreter for this model and allocate its tensors.

    Returns:
        tuple:
            - tens (list[dict]): The raw tensor detail dictionaries returned by
              tf.lite.Interpreter.get_tensor_details(). Each entry describes a tensor
              (index, name, shape, dtype, etc.).
            - const_flags (dict[int, bool]): A mapping from tensor index to a boolean flag
              indicating whether the tensor is inferred to be a constant buffer (True) or
              a runtime tensor/activation (False).
              
    Notes:
        - The heuristics are conservative and may not be 100% accurate for all TFLite models.
        - Use the returned const_flags to separate model parameters/buffers from activations when
          analyzing or visualizing model internals.
    """
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    tens = interp.get_tensor_details()
    # Mark constants (buffers) vs runtime tensors
    const_flags = {}
    for t in tens:
        # TensorFlow marks constant tensors with 'buffer' backing; in python API we infer:
        # If 'quantization_parameters' exists -> still could be const or activation. We'll re-check by name heuristics.
        const_flags[t['index']] = ('Const' in t['name']) or ('/Const' in t['name']) or (t.get('sparsity_parameters') is not None)
    return tens, const_flags