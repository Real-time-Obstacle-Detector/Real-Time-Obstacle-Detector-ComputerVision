import collections
from utils import conv2d_macs, depthwise_macs

def estimate_macs_from_analyzer(analyzer_text):
    """
    Parse ops and shapes from Analyzer dump (heuristic).
    Returns dict: {'total_macs': ..., 'by_op': Counter, 'by_section': Counter}
    Section labeling is heuristic; adjust regexes based on your naming (backbone/neck/head/exit).
    """
    by_op = collections.Counter()
    by_section = collections.Counter()
    total = 0

    lines = analyzer_text.splitlines()
    cur_op = None
    cur_name = ""
    in_shape = None
    w_shape = None

    for ln in lines:
        if "OpCode" in ln and "builtin_code" in ln:
            cur_op = ln.split("builtin_code:")[1].strip().split()[0]
        if "Operator" in ln and "name:" in ln:
            cur_name = ln.split("name:")[1].strip()
        if "inputs: [" in ln and "shape:" in ln:
            # Try to read an activation shape like shape: [1, H, W, C]
            pass
        # Simple shape captures:
        m_in = None
        if "input_tensors: [" in ln and "shape: [" in ln:
            m_in = ln.split("shape: [")[-1].split("]")[0]
            try:
                in_shape = [int(x.strip()) for x in m_in.split(",")]
            except:
                in_shape = None
        if "weights" in ln and "shape: [" in ln:
            m_w = ln.split("shape: [")[-1].split("]")[0]
            try:
                w_shape = [int(x.strip()) for x in m_w.split(",")]
            except:
                w_shape = None

        if "outputs:" in ln:
            # We have enough info for some ops:
            macs = 0
            if cur_op in ("CONV_2D", "CONV_2D_REF") and in_shape and w_shape and len(in_shape)==4 and len(w_shape)==4:
                n,h,w,c = in_shape
                kh, kw, cin, cout = w_shape  # TFLite uses [kh, kw, Cin, Cout]
                macs = conv2d_macs(h,w,cin,cout,kh,kw)
            elif cur_op in ("DEPTHWISE_CONV_2D",):
                if in_shape and w_shape and len(in_shape)==4 and len(w_shape)==4:
                    n,h,w,c = in_shape
                    kh, kw, ch_mult, cin = w_shape[0], w_shape[1], w_shape[3], in_shape[3]
                    macs = depthwise_macs(h,w,cin,kh,kw)
            elif cur_op in ("FULLY_CONNECTED",):
                # If shapes known, macs ~ in_dim*out_dim
                pass

            total += macs
            by_op[cur_op] += macs

            # Heuristic section labeling from names (adapt to your naming):
            sec = "unknown"
            s = cur_name.lower()
            if "backbone" in s: sec = "backbone"
            elif "neck" in s or "pan" in s or "fpn" in s: sec = "neck"
            elif "head" in s or "detect" in s: sec = "head"
            elif "exit" in s: sec = "early_exit_head"
            by_section[sec] += macs

            # Reset for next op
            cur_op, cur_name, in_shape, w_shape = None, "", None, None

    return {'total_macs': total, 'by_op': dict(by_op), 'by_section': dict(by_section)}

