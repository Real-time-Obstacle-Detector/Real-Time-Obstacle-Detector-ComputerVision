import re, collections


def parse_param_counts_from_analyzer(text):
    """
    Parses tensor parameter counts from an Analyzer dump text.

    The function uses a regular expression to extract tensor information from the provided text,
    which is expected to contain lines formatted like:
    "Tensor[<idx>]{name: <name>, type: <dtype>, shape: [<shape>]}"
    It computes the total number of parameters, counts per data type, and details for each tensor.

    Args:
        text (str): The Analyzer dump text containing tensor information.

    Returns:
        tuple:
            params_total (int): Total number of parameters across all tensors.
            per_dtype (collections.Counter): Counter of parameter counts per data type.
            per_tensor (list): List of tuples containing tensor details:
                (index (int), name (str), dtype (str), shape (tuple of int), param_count (int))
    """
    tensor_re = re.compile(r"Tensor\[(\d+)\]\{name: (.*?), type: (.*?), shape: \[(.*?)\]")
    params_total = 0
    per_dtype = collections.Counter()
    per_tensor = []
    for m in tensor_re.finditer(text):
        idx, name, dtype, shape_txt = m.groups()
        shape = [int(s) for s in shape_txt.split(",") if s.strip()] if shape_txt.strip() else []
        n = 1
        for d in shape:
            n *= d
        # count as params if this tensor is constant (we'll infer later via Interpreter)
        per_tensor.append((int(idx), name, dtype, tuple(shape), n))
        per_dtype[dtype] += n
        params_total += n
    return params_total, per_dtype, per_tensor