import io
import contextlib
import tensorflow as tf

def dump_analyzer(model_path=None, model_bytes=None, gpu_compatibility=False):
    """
    Returns the TFLite analyzer text as a Python string by capturing console output.
    """
    if not model_path and not model_bytes:
        raise ValueError("Provide model_path or model_bytes")

    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        tf.lite.experimental.Analyzer.analyze(
            model_path=model_path,
            model_content=model_bytes,
            gpu_compatibility=gpu_compatibility
        )

    text = out.getvalue().strip()
    if not text:
        text = err.getvalue().strip()

    if not text:
        raise RuntimeError("Analyzer produced no output. "
                           "Verify TensorFlow version and model path.")

    return text